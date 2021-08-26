### edge_correction_weights.py (Oliver Philcox, 2021)
# Compute the edge coerrection matrices for the NPCF basis functions
# These are saved as .npy files which can be read in by the combine_files.py code

import sys, os
import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j, wigner_9j

## First read-in the input parameters from the command line
if len(sys.argv)!=4:
    raise Exception("Need to specify N, LMAX and N_threads")
else:
    N = int(sys.argv[1])
    LMAX = int(sys.argv[2])
    threads = int(sys.argv[3])

print("\nComputing the edge-correction matrices for the %dPCF up to l_max = %d\n"%(N,LMAX))

if N==3:
    # First define array of ells
    ell_1 = [ell for ell in range(0,LMAX+1,1)]

    # Now compute coupling matrix
    print("Computing 3PCF coupling matrix")
    coupling_matrix = np.zeros((len(ell_1),len(ell_1),len(ell_1)))
    for l_i in ell_1:
        for l_j in ell_1:
            for l_inner in ell_1:
                xx = np.float64(1./(4.*np.pi)*wigner_3j(l_i,l_inner,l_j,0,0,0)**2*np.sqrt(2.*l_i+1.)*np.sqrt(2.*l_j+1.)*np.sqrt(2.*l_inner+1.))
                coupling_matrix[l_i,l_j,l_inner] = xx

elif N==4:
    # First define array of ells
    ell_1,ell_2,ell_3 = [[] for _ in range(3)]
    for l1 in range(0,LMAX+1,1):
        for l2 in range(0,LMAX+1,1):
            for l3 in range(abs(l1-l2),min(l1+l2,LMAX)+1,1):
                if (-1.)**(l1+l2+l3)==-1: continue
                ell_1.append(l1)
                ell_2.append(l2)
                ell_3.append(l3)

    # Define coupling matrix, by iterating over all Lambda triples
    print("Computing 4PCF coupling matrix on %d CPUs"%threads)

    def compute_matrix_coeff(i):
        # output submatrix
        tmp_out = np.zeros((len(ell_1),len(ell_1)))

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

                # Compute 3j couplings
                three_j_piece = np.float64(wigner_3j(L_1,Lp_1,Lpp_1,0,0,0)*wigner_3j(L_2,Lp_2,Lpp_2,0,0,0)*wigner_3j(L_3,Lp_3,Lpp_3,0,0,0))
                if three_j_piece==0: continue

                # Compute the 9j component
                nine_j_piece = np.float64(wigner_9j(L_1,Lp_1,Lpp_1,L_2,Lp_2,Lpp_2,L_3,Lp_3,Lpp_3,prec=8))
                if nine_j_piece==0: continue

                tmp_out[j,k] = pref * three_j_piece * nine_j_piece
        return tmp_out

    pool = multiprocessing.Pool(threads)
    coupling_matrix = np.asarray(list(pool.map(compute_matrix_coeff, range(len(ell_1)))))

elif N==5:
    # First define array of ells
    ell_1,ell_2,ell_12,ell_3,ell_4 = [[] for _ in range(5)]
    for l1 in range(0,LMAX+1,1):
        for l2 in range(0,LMAX+1,1):
            for l12 in range(abs(l1-l2),l1+l2+1,1):
                for l3 in range(0,LMAX+1,1):
                    for l4 in range(abs(l12-l3),min(l12+l3,LMAX)+1,1):
                        if (-1.)**(l1+l2+l3+l4)==-1: continue
                        ell_1.append(l1)
                        ell_2.append(l2)
                        ell_12.append(l12)
                        ell_3.append(l3)
                        ell_4.append(l4)

    # Define coupling matrix, by iterating over all Lambda triples
    print("Computing 5PCF coupling matrix on %d CPUs"%threads)

    def compute_matrix_coeff(i):
        # output submatrix
        tmp_out = np.zeros((len(ell_1),len(ell_1)))

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

                # Compute 3j couplings
                three_j_piece = np.float64(wigner_3j(L_1,Lp_1,Lpp_1,0,0,0)*wigner_3j(L_2,Lp_2,Lpp_2,0,0,0)*wigner_3j(L_3,Lp_3,Lpp_3,0,0,0)*wigner_3j(L_4,Lp_4,Lpp_4,0,0,0))
                if three_j_piece==0: continue

                # Compute the 9j component
                nine_j_piece = np.float64(wigner_9j(L_1,L_2,L_12,Lp_1,Lp_2,Lp_12,Lpp_1,Lpp_2,Lpp_12,prec=8))*np.float64(wigner_9j(L_12,L_3,L_4,Lp_12,Lp_3,Lp_4,Lpp_12,Lpp_3,Lpp_4,prec=8))
                if nine_j_piece==0: continue

                tmp_out[j,k] = pref * three_j_piece * nine_j_piece
        return tmp_out

    pool = multiprocessing.Pool(threads)
    coupling_matrix = np.asarray(list(pool.map(compute_matrix_coeff, range(len(ell_1)))))

elif N==6:
    # First define array of ells
    ell_1,ell_2,ell_12,ell_3,ell_123,ell_4,ell_5 = [[] for _ in range(7)]
    for l1 in range(0,LMAX+1,1):
        for l2 in range(0,LMAX+1,1):
            for l12 in range(abs(l1-l2),l1+l2+1,1):
                for l3 in range(0,LMAX+1,1):
                    for l123 in range(abs(l12-l3),l12+l3+1,1):
                        for l4 in range(0,LMAX+1,1):
                            for l5 in range(abs(l123-l4),min(l123+l4,LMAX)+1,1):
                                if (-1.)**(l1+l2+l3+l4+l5)==-1: continue
                                ell_1.append(l1)
                                ell_2.append(l2)
                                ell_12.append(l12)
                                ell_3.append(l3)
                                ell_123.append(l123)
                                ell_4.append(l4)
                                ell_5.append(l5)

    # Define coupling matrix, by iterating over all Lambda triples
    print("Computing 6PCF coupling matrix on %d CPUs"%threads)

    def compute_matrix_coeff(i):
        # output submatrix
        tmp_out = np.zeros((len(ell_1),len(ell_1)))

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
                if three_j_piece==0: continue

                # Compute the 9j component
                nine_j_piece = np.float64(wigner_9j(L_1,L_2,L_12,Lp_1,Lp_2,Lp_12,Lpp_1,Lpp_2,Lpp_12,prec=8))*np.float64(wigner_9j(L_12,L_3,L_123,Lp_12,Lp_3,Lp_123,Lpp_12,Lpp_3,Lpp_123,prec=8))*np.float64(wigner_9j(L_123,L_4,L_5,Lp_123,Lp_4,Lp_5,Lpp_123,Lpp_4,Lpp_5,prec=8))
                if nine_j_piece==0: continue

                tmp_out[j,k] = pref * three_j_piece * nine_j_piece
        return tmp_out

    pool = multiprocessing.Pool(threads)
    coupling_matrix = np.asarray(list(pool.map(compute_matrix_coeff, range(len(ell_1)))))

outfile = os.path.dirname(os.path.realpath(sys.argv[0]))+'/coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d.npy'%(N,LMAX)
print(outfile)
np.save(outfile,coupling_matrix)
print("Coupling matrix saved to %s"%outfile)
