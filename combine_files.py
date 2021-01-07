### combine_files.py (Oliver Philcox, 2021)
# This reads in a set of (data-random) and (random) particle counts and uses them to construct the N-point functions, including edge-correction
# It is designed to be used with the run_npcf.csh script
# Currently 3PCF, 4PCF, 5PCF and 6PCF are supported, and multithreading is used to speed up the 4PCF, 5PCF + 6PCF coupling matrix computations
# The output is saved to the working directory with the same format as the NPCF counts, with the filename ...zeta_{N}pcf.txt

import sys, os
import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j, wigner_9j

## First read-in the input file string from the command line
if len(sys.argv)!=3:
    raise Exception("Need to specify the input files and N_threads!")
else:
    inputs = str(sys.argv[1])
    threads = int(sys.argv[2])

print("Reading in files starting with %s\n"%inputs)

# Decide which N we're using
Ns = []
for N in range(10):
    R_file = inputs+'.r_%dpcf.txt'%N
    if os.path.exists(R_file):
        Ns.append(N)

for N in Ns:
    # First load in R piece
    R_file = inputs+'.r_%dpcf.txt'%N
    countsR = np.loadtxt(R_file,skiprows=4+N) # skipping rows with radial bins

    # Extract ells and radial bins
    if N==3:
        ell_1 = np.asarray(countsR[:,0],dtype=int)
        max_ell = np.max(ell_1)
        countsR = countsR[:,1:]
        bin1,bin2 = np.loadtxt(R_file,skiprows=5,max_rows=2)
    elif N==4:
        ell_1,ell_2,ell_3 = np.asarray(countsR[:,:3],dtype=int).T
        max_ell = np.max(ell_1)
        countsR = countsR[:,3:]
        bin1,bin2,bin3 = np.loadtxt(R_file,skiprows=5,max_rows=3)
    elif N==5:
        ell_1,ell_2,ell_12,ell_3,ell_4 = np.asarray(countsR[:,:5],dtype=int).T
        max_ell = np.max(ell_1)
        countsR = countsR[:,5:]
        bin1,bin2,bin3,bin4 = np.loadtxt(R_file,skiprows=5,max_rows=4)
    elif N==6:
        ell_1,ell_2,ell_12,ell_3,ell_123,ell_4,ell_5 = np.asarray(countsR[:,:7],dtype=int).T
        max_ell = np.max(ell_1)
        countsR = countsR[:,7:]
        bin1,bin2,bin3,bin4,bin5 = np.loadtxt(R_file,skiprows=5,max_rows=5)
    else:
        raise Exception("%dPCF not yet configured"%N)

    # Now load in D-R pieces and average
    countsN_all = []
    total_DmR = 0
    for i in range(100):
        DmR_file = inputs+'.n%s_%dpcf.txt'%(str(i).zfill(2),N)
        if not os.path.exists(DmR_file): continue
        # Extract counts
        if N==3:
            countsN_all.append(np.loadtxt(DmR_file,skiprows=7)[:,1:]) # skipping rows with radial bins and ell
        elif N==4:
            countsN_all.append(np.loadtxt(DmR_file,skiprows=8)[:,3:])
        elif N==5:
            countsN_all.append(np.loadtxt(DmR_file,skiprows=9)[:,5:])
        elif N==6:
            countsN_all.append(np.loadtxt(DmR_file,skiprows=10)[:,7:])
    countsN_all = np.asarray(countsN_all)
    N_files = len(countsN_all)
    countsN = np.mean(countsN_all,axis=0)
    # could use this next line to compute std from finite number of randoms!
    #countsNsig = np.std(countsN_all,axis=0)/np.sqrt(N_files)

    # Now compute edge-correction equations

    if N==3:
        # Define coupling coefficients, rescaling by R_{ell=0}
        assert ell_1[0]==0
        f_ell = countsR/countsR[0] # (first row should be unity!)

        # Define coupling matrix
        print("Computing 3PCF coupling matrix")
        coupling_matrix = np.zeros((len(ell_1),len(ell_1),len(bin1)))
        for l_i in ell_1:
            for l_j in ell_1:
                for l_inner in ell_1:
                    xx = np.float64(1./(4.*np.pi)*wigner_3j(l_i,l_inner,l_j,0,0,0)**2*f_ell[l_inner]*np.sqrt(2.*l_i+1.)*np.sqrt(2.*l_j+1.)*np.sqrt(2.*l_inner+1.))
                    coupling_matrix[l_i,l_j] += xx

        ## Now invert matrix equation to get zeta
        # Note that our matrix definition is symmetric
        zeta = np.zeros_like(countsN)
        for i in range(len(bin1)):
            zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

        # Now save the output to file, copying the first few lines from the N files
        zeta_file = inputs+'.zeta_%dpcf.txt'%N
        rfile = open(R_file,"r")
        zfile = open(zeta_file,"w")
        for l,line in enumerate(rfile):
            if l>=7: continue
            zfile.write(line)
        for a in range(len(zeta)):
            zfile.write("%d\t"%ell_1[a])
            for b in range(len(zeta[a])):
                zfile.write("%.8e\t"%zeta[a,b])
            zfile.write("\n")
        zfile.close()

    if N==4:
        # Define coupling coefficients, rescaling by R_{Lambda=0}
        assert ell_1[0]==ell_2[0]==ell_3[0]
        f_Lambda = countsR/countsR[0] # (first row should be unity!)

        # Define coupling matrix, by iterating over all Lambda triples
        print("Computing 4PCF coupling matrix on %d CPUs"%threads)

        def compute_matrix_coeff(i):
            # output submatrix
            tmp_out = np.zeros((len(ell_1),len(bin1)))

            # i is first matrix index
            L_1,L_2,L_3=ell_1[i],ell_2[i],ell_3[i]
            pref_1 = np.sqrt((2.*L_1+1.)*(2.*L_2+1.)*(2.*L_3+1.))

            for j in range(len(ell_1)):
                # j is second matrix index
                Lpp_1,Lpp_2,Lpp_3=ell_1[j],ell_2[j],ell_3[j]
                pref_2 = pref_1*np.sqrt((2.*Lpp_1+1.)*(2.*Lpp_2+1.)*(2.*Lpp_3+1.))

                for k in range(len(ell_1)):
                    # k indexes inner Lambda' term
                    Lp_1,Lp_2,Lp_3 = ell_1[k],ell_2[k],ell_3[k]

                    # Compute prefactor
                    pref = pref_2*np.sqrt((2.*Lp_1+1.)*(2.*Lp_2+1.)*(2.*Lp_3+1.))/(4.*np.pi)**(3./2.)

                    # Compute three-J couplings
                    three_j_piece = np.float64(wigner_3j(L_1,Lp_1,Lpp_1,0,0,0)*wigner_3j(L_2,Lp_2,Lpp_2,0,0,0)*wigner_3j(L_3,Lp_3,Lpp_3,0,0,0))

                    # Compute the 9j component
                    nine_j_piece = np.float64(wigner_9j(L_1,Lp_1,Lpp_1,L_2,Lp_2,Lpp_2,L_3,Lp_3,Lpp_3,prec=8))

                    tmp_out[j] += pref * three_j_piece * nine_j_piece * f_Lambda[k]
            return tmp_out

        pool = multiprocessing.Pool(threads)
        coupling_matrix = np.asarray(list(pool.map(compute_matrix_coeff, range(len(ell_1)))))
        print("Coupling matrix computed")

        ## Now invert matrix equation to get zeta
        # Note that our matrix definition is symmetric
        zeta = np.zeros_like(countsN)
        for i in range(len(bin1)):
            zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

        # Now save the output to file, copying the first few lines from the N files
        zeta_file = inputs+'.zeta_%dpcf.txt'%N
        rfile = open(R_file,"r")
        zfile = open(zeta_file,"w")
        for l,line in enumerate(rfile):
            if l>=8: continue
            zfile.write(line)
        for a in range(len(zeta)):
            zfile.write("%d\t"%ell_1[a])
            zfile.write("%d\t"%ell_2[a])
            zfile.write("%d\t"%ell_3[a])
            for b in range(len(zeta[a])):
                zfile.write("%.8e\t"%zeta[a,b])
            zfile.write("\n")
        zfile.close()

    if N==5:
        # Define coupling coefficients, rescaling by R_{Lambda=0}
        assert ell_1[0]==ell_2[0]==ell_3[0]==ell_12[0]==ell_4[0]
        f_Lambda = countsR/countsR[0] # (first row should be unity!)

        # Define coupling matrix, by iterating over all Lambda triples
        print("Computing 5PCF coupling matrix on %d CPUs"%threads)

        def compute_matrix_coeff(i):
            # output submatrix
            tmp_out = np.zeros((len(ell_1),len(bin1)))

            # i is first matrix index
            L_1,L_2,L_12,L_3,L_4=ell_1[i],ell_2[i],ell_12[i],ell_3[i],ell_4[i]

            pref_1 = np.sqrt((2.*L_1+1.)*(2.*L_2+1.)*(2.*L_12+1.)*(2.*L_3+1.)*(2.*L_4+1.))

            for j in range(len(ell_1)):
                # j is second matrix index
                Lpp_1,Lpp_2,Lpp_12,Lpp_3,Lpp_4=ell_1[j],ell_2[j],ell_12[j],ell_3[j],ell_4[j]

                pref_2 = pref_1*np.sqrt((2.*Lpp_1+1.)*(2.*Lpp_2+1.)*(2.*Lpp_12+1.)*(2.*Lpp_3+1.)*(2.*Lpp_4+1.))

                for k in range(len(ell_1)):
                    # k indexes inner Lambda' term
                    Lp_1,Lp_2,Lp_12,Lp_3,Lp_4 = ell_1[k],ell_2[k],ell_12[k],ell_3[k],ell_4[k]

                    # Compute prefactor
                    pref = pref_2*np.sqrt((2.*Lp_1+1.)*(2.*Lp_2+1.)*(2.*Lp_12+1.)*(2.*Lp_3+1.)*(2.+Lp_4+1.))/(4.*np.pi)**2.

                    # Compute three-J couplings
                    three_j_piece = np.float64(wigner_3j(L_1,Lp_1,Lpp_1,0,0,0)*wigner_3j(L_2,Lp_2,Lpp_2,0,0,0)*wigner_3j(L_3,Lp_3,Lpp_3,0,0,0)*wigner_3j(L_4,Lp_4,Lpp_4,0,0,0))

                    # Compute the 9j component
                    nine_j_piece = np.float64(wigner_9j(L_1,L_2,L_12,Lp_1,Lp_2,Lp_12,Lpp_1,Lpp_2,Lpp_12,prec=8))*np.float64(wigner_9j(L_12,L_3,L_4,Lp_12,Lp_3,Lp_4,Lpp_12,Lpp_3,Lpp_4,prec=8))

                    tmp_out[j] += pref * three_j_piece * nine_j_piece * f_Lambda[k]
            return tmp_out

        pool = multiprocessing.Pool(threads)
        coupling_matrix = np.asarray(list(pool.map(compute_matrix_coeff, range(len(ell_1)))))
        print("Coupling matrix computed")

        ## Now invert matrix equation to get zeta
        # Note that our matrix definition is symmetric
        zeta = np.zeros_like(countsN)
        for i in range(len(bin1)):
            zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

        # Now save the output to file, copying the first few lines from the N files
        zeta_file = inputs+'.zeta_%dpcf.txt'%N
        rfile = open(R_file,"r")
        zfile = open(zeta_file,"w")
        for l,line in enumerate(rfile):
            if l>=9: continue
            zfile.write(line)
        for a in range(len(zeta)):
            zfile.write("%d\t"%ell_1[a])
            zfile.write("%d\t"%ell_2[a])
            zfile.write("%d\t"%ell_12[a])
            zfile.write("%d\t"%ell_3[a])
            zfile.write("%d\t"%ell_4[a])
            for b in range(len(zeta[a])):
                zfile.write("%.8e\t"%zeta[a,b])
            zfile.write("\n")
        zfile.close()

    if N==6:
        # Define coupling coefficients, rescaling by R_{Lambda=0}
        assert ell_1[0]==ell_2[0]==ell_12[0]==ell_3[0]==ell_123[0]==ell_4[0]==ell_5[0]
        f_Lambda = countsR/countsR[0] # (first row should be unity!)

        # Define coupling matrix, by iterating over all Lambda triples
        print("Computing 6PCF coupling matrix on %d CPUs"%threads)

        def compute_matrix_coeff(i):
            print("Computing coefficient %d of %d"%(i+1,len(ell_1)))
            # output submatrix
            tmp_out = np.zeros((len(ell_1),len(bin1)))

            # i is first matrix index
            L_1,L_2,L_12,L_3,L_123,L_4,L_5=ell_1[i],ell_2[i],ell_12[i],ell_3[i],ell_123[i],ell_4[i],ell_5[i]

            pref_1 = np.sqrt((2.*L_1+1.)*(2.*L_2+1.)*(2.*L_12+1.)*(2.*L_3+1.)*(2.*L_123+1.)*(2.*L_4+1.)*(2.*L_5+1.))

            for j in range(len(ell_1)):
                # j is second matrix index
                Lpp_1,Lpp_2,Lpp_12,Lpp_3,Lpp_123,Lpp_4,Lpp_5=ell_1[j],ell_2[j],ell_12[j],ell_3[j],ell_123[j],ell_4[j],ell_5[j]

                pref_2 = pref_1*np.sqrt((2.*Lpp_1+1.)*(2.*Lpp_2+1.)*(2.*Lpp_12+1.)*(2.*Lpp_3+1.)*(2.*Lpp_123)*(2.*Lpp_4+1.)*(2.*Lpp_5+1.))

                for k in range(len(ell_1)):
                    # k indexes inner Lambda' term
                    Lp_1,Lp_2,Lp_12,Lp_3,Lp_123,Lp_4,Lp_5 = ell_1[k],ell_2[k],ell_12[k],ell_3[k],ell_123[k],ell_4[k],ell_5[k]

                    # Compute prefactor
                    pref = pref_2*np.sqrt((2.*Lp_1+1.)*(2.*Lp_2+1.)*(2.*Lp_12+1.)*(2.*Lp_3+1.)*(2.*Lp_123+1.)*(2.+Lp_4+1.)*(2.+Lp_5+1.))/(4.*np.pi)**(5./2.)

                    # Compute three-J couplings
                    three_j_piece = np.float64(wigner_3j(L_1,Lp_1,Lpp_1,0,0,0)*wigner_3j(L_2,Lp_2,Lpp_2,0,0,0)*wigner_3j(L_3,Lp_3,Lpp_3,0,0,0)*wigner_3j(L_4,Lp_4,Lpp_4,0,0,0)*wigner_3j(L_5,Lp_5,Lpp_5,0,0,0))

                    # Compute the 9j component
                    nine_j_piece = np.float64(wigner_9j(L_1,L_2,L_12,Lp_1,Lp_2,Lp_12,Lpp_1,Lpp_2,Lpp_12,prec=8))*np.float64(wigner_9j(L_12,L_3,L_123,Lp_12,Lp_3,Lp_123,Lpp_12,Lpp_3,Lpp_123,prec=8))*np.float64(wigner_9j(L_123,L_4,L_5,Lp_123,Lp_4,Lp_5,Lpp_123,Lpp_4,Lpp_5,prec=8))

                    tmp_out[j] += pref * three_j_piece * nine_j_piece * f_Lambda[k]
            return tmp_out

        pool = multiprocessing.Pool(threads)
        coupling_matrix = np.asarray(list(pool.map(compute_matrix_coeff, range(len(ell_1)))))
        print("Coupling matrix computed")

        ## Now invert matrix equation to get zeta
        # Note that our matrix definition is symmetric
        print("Coupling matrix computed")
        zeta = np.zeros_like(countsN)
        for i in range(len(bin1)):
            zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

        # Now save the output to file, copying the first few lines from the N files
        zeta_file = inputs+'.zeta_%dpcf.txt'%N
        rfile = open(R_file,"r")
        zfile = open(zeta_file,"w")
        for l,line in enumerate(rfile):
            if l>=10: continue
            zfile.write(line)
        for a in range(len(zeta)):
            zfile.write("%d\t"%ell_1[a])
            zfile.write("%d\t"%ell_2[a])
            zfile.write("%d\t"%ell_12[a])
            zfile.write("%d\t"%ell_3[a])
            zfile.write("%d\t"%ell_123[a])
            zfile.write("%d\t"%ell_4[a])
            zfile.write("%d\t"%ell_5[a])
            for b in range(len(zeta[a])):
                zfile.write("%.8e\t"%zeta[a,b])
            zfile.write("\n")
        zfile.close()

    print("Computed %dPCF using %d (data-random) files, saving to %s\n"%(N,N_files,zeta_file))
