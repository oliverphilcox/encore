### combine_files_periodic.py (Oliver Philcox, 2021)
# This reads in a set of (data-random) particle counts and uses them to construct the N-point functions, assuming a periodic geometry
# It is designed to be used with the run_npcf.csh script
# Currently 3PCF, 4PCF, 5PCF and 6PCF are supported, and the R^N random counts are computed analytically.
# The output is saved to the working directory with the same format as the NPCF counts, with the filename ...zeta_{N}pcf.txt

import sys, os
import numpy as np

## First read-in the input file string from the command line
if len(sys.argv)!=5:
    raise Exception("Need to specify the input files, number of galaxies, boxsize and R_max!")
else:
    inputs = str(sys.argv[1])
    N_gal = int(sys.argv[2])
    boxsize = np.float64(sys.argv[3])
    R_max = np.float64(sys.argv[4])
n_bar = np.float64(N_gal)/boxsize**3.

print("Reading in files starting with %s\n"%inputs)

# Decide which N we're using
Ns = []
for N in range(10):
    R_file = inputs+'.r_%dpcf.txt'%N
    if os.path.exists(R_file):
        Ns.append(N)

for N in Ns:
    # Load in first D-R piece to get bins
    tmp_file = inputs+'.n00_%dpcf.txt'%(N)
    if N==2:
        counts_tmp = np.loadtxt(tmp_file,skiprows=4) # skipping rows with radial bins
    else:
        counts_tmp = np.loadtxt(tmp_file,skiprows=4+N) # skipping rows with radial bins

    # Extract ells and radial bins
    if N==2:
        bin1 = np.loadtxt(tmp_file,skiprows=3,max_rows=1)
    elif N==3:
        ell_1 = np.asarray(counts_tmp[:,0],dtype=int)
        max_ell = np.max(ell_1)
        bin1,bin2 = np.loadtxt(tmp_file,skiprows=5,max_rows=2)
    elif N==4:
        ell_1,ell_2,ell_3 = np.asarray(counts_tmp[:,:3],dtype=int).T
        max_ell = np.max(ell_1)
        bin1,bin2,bin3 = np.loadtxt(tmp_file,skiprows=5,max_rows=3)
    elif N==5:
        ell_1,ell_2,ell_12,ell_3,ell_4 = np.asarray(counts_tmp[:,:5],dtype=int).T
        max_ell = np.max(ell_1)
        bin1,bin2,bin3,bin4 = np.loadtxt(tmp_file,skiprows=5,max_rows=4)
    elif N==6:
        ell_1,ell_2,ell_12,ell_3,ell_123,ell_4,ell_5 = np.asarray(counts_tmp[:,:7],dtype=int).T
        max_ell = np.max(ell_1)
        bin1,bin2,bin3,bin4,bin5 = np.loadtxt(tmp_file,skiprows=5,max_rows=5)
    else:
        raise Exception("%dPCF not yet configured"%N)

    # Now load in D-R pieces and average
    countsN_all = []
    total_DmR = 0
    for i in range(100):
        DmR_file = inputs+'.n%s_%dpcf.txt'%(str(i).zfill(2),N)
        if not os.path.exists(DmR_file): continue
        # Extract counts
        if N==2:
            countsN_all.append(np.loadtxt(DmR_file,skiprows=4))
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

    ### Now compute analytic randoms
    ### For N>2 we include the basis function P_{0} = (4\pi)^{-(N-1)/2}
    if N==2:
        n_r = 1.*len(bin1)
        nV1 = n_bar*4.*np.pi/3.*(((bin1+1)*R_max/n_r)**3.-(bin1*R_max/n_r)**3.)
        R_analyt = N_gal*nV1
    elif N==3:
        n_r = len(np.unique(np.concatenate([bin1,bin2])))
        nV1 = n_bar*4.*np.pi/3.*(((bin1+1)*R_max/n_r)**3.-(bin1*R_max/n_r)**3.)
        nV2 = n_bar*4.*np.pi/3.*(((bin2+1)*R_max/n_r)**3.-(bin2*R_max/n_r)**3.)
        R_analyt = N_gal*nV1*nV2/(4.*np.pi)
    elif N==4:
        n_r = len(np.unique(np.concatenate([bin1,bin2,bin3])))
        nV1 = n_bar*4.*np.pi/3.*(((bin1+1)*R_max/n_r)**3.-(bin1*R_max/n_r)**3.)
        nV2 = n_bar*4.*np.pi/3.*(((bin2+1)*R_max/n_r)**3.-(bin2*R_max/n_r)**3.)
        nV3 = n_bar*4.*np.pi/3.*(((bin3+1)*R_max/n_r)**3.-(bin3*R_max/n_r)**3.)
        R_analyt = N_gal*nV1*nV2*nV3/(4.*np.pi)**1.5
    elif N==5:
        n_r = len(np.unique(np.concatenate([bin1,bin2,bin3,bin4])))
        nV1 = n_bar*4.*np.pi/3.*(((bin1+1)*R_max/n_r)**3.-(bin1*R_max/n_r)**3.)
        nV2 = n_bar*4.*np.pi/3.*(((bin2+1)*R_max/n_r)**3.-(bin2*R_max/n_r)**3.)
        nV3 = n_bar*4.*np.pi/3.*(((bin3+1)*R_max/n_r)**3.-(bin3*R_max/n_r)**3.)
        nV4 = n_bar*4.*np.pi/3.*(((bin4+1)*R_max/n_r)**3.-(bin4*R_max/n_r)**3.)
        R_analyt = N_gal*nV1*nV2*nV3*nV4/(4.*np.pi)**2.
    elif N==6:
        n_r = len(np.unique(np.concatenate([bin1,bin2,bin3,bin4,bin5])))
        nV1 = n_bar*4.*np.pi/3.*(((bin1+1)*R_max/n_r)**3.-(bin1*R_max/n_r)**3.)
        nV2 = n_bar*4.*np.pi/3.*(((bin2+1)*R_max/n_r)**3.-(bin2*R_max/n_r)**3.)
        nV3 = n_bar*4.*np.pi/3.*(((bin3+1)*R_max/n_r)**3.-(bin3*R_max/n_r)**3.)
        nV4 = n_bar*4.*np.pi/3.*(((bin4+1)*R_max/n_r)**3.-(bin4*R_max/n_r)**3.)
        nV5 = n_bar*4.*np.pi/3.*(((bin5+1)*R_max/n_r)**3.-(bin5*R_max/n_r)**3.)
        R_analyt = N_gal*nV1*nV2*nV3*nV4*nV5/(4.*np.pi)**2.5

    # Now compute full NPCF
    # We do not apply any edge-correction, such that zeta = (D-R)_Lambda / R_{Lambda=0} * (4 pi)^{(N-1)/2}
    # i.e. we assume R_{Lambda} = 0 for Lambda > (00...0)
    # the (4 pi) factor comes from the P_{0} basis function.

    if N==2:
        # no (4 pi) factors here since no angular basis functions!
        zeta = countsN/R_analyt
    else:
        zeta = np.zeros_like(countsN)
        for i in range(len(bin1)):
            zeta[:,i] = countsN[:,i]/R_analyt[i]*(4.*np.pi)**(0.5*(N-1.))

    # Now save the output to file, copying the first few lines from the N files
    zeta_file = inputs+'.zeta_%dpcf.txt'%N
    rfile = open(tmp_file,"r")
    zfile = open(zeta_file,"w")
    if N==2:
        for l,line in enumerate(rfile):
            if l>=4: continue
            zfile.write(line)
        for a in range(len(zeta)):
            zfile.write("%.8e\t"%zeta[a])
    else:
        for l,line in enumerate(rfile):
            if l>=4+N: continue
            zfile.write(line)
        for a in range(len(zeta)):
            zfile.write("%d\t"%ell_1[a])
            for b in range(len(zeta[a])):
                zfile.write("%.8e\t"%zeta[a,b])
            zfile.write("\n")
    zfile.close()

    print("Computed periodic %dPCF using %d (data-random) files, saving to %s\n"%(N,N_files,zeta_file))
