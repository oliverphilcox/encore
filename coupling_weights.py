### coupling_weights.py (Oliver Philcox, 2021)
# Compute the coupling coefficients C_m^Lambda for the NPCF basis functions
# These are saved in a .txt file as a one-dimensional array and will be read in by the C++ code.
# Input parameters are N and LMAX (with 3<=N<=6).
# We caution that the computation time scales as (LMAX+1)^{3N-7}, so large LMAX should be avoided for high N.
# Note that we save only the parity-even coefficients (with even l1+l2+...+lN)

import sys, os
import numpy as np
from sympy.physics.wigner import wigner_3j

## First read-in the input parameters from the command line
if len(sys.argv)!=3:
    raise Exception("Need to specify N and LMAX")
else:
    N = int(sys.argv[1])
    LMAX = int(sys.argv[2])

print("\nComputing the coupling coefficients for the %dPCF up to l_max = %d\n"%(N,LMAX))

if N==3:
    # 3PCF computation
    def WeightFunction(l,m):
        """C_m^Lambda for 2nd order isotropic basis function"""
        return pow(-1.,l1-m1)/np.sqrt(2.*l1+1.)

    WeightsNPCF = np.zeros(int((LMAX+1)*(LMAX+2)/2.))
    for l1 in range(LMAX+1):
        for m1 in range(l1+1):
            weight = WeightFunction(l1,m1)
            if weight!=0:
                WeightsNPCF[l1*(l1+1)//2+m1] = weight

if N==4:
    # 4PCF computation
    def WeightFunction(l1,l2,l3,m1,m2,m3):
        """C_m^Lambda for 3rd order isotropic basis function"""
        return wigner_3j(l1,l2,l3,m1,m2,m3)*pow(-1.,l1+l2+l3)

    # Create 1D output matrix. The size is an overestimate and will be trimmed down later.
    WeightsNPCF = np.zeros(((LMAX+1)**5))
    index = 0
    for l1 in range(LMAX+1):
        for l2 in range(LMAX+1):
            for l3 in range(abs(l1-l2),min(LMAX,l1+l2)+1):
                if pow(-1.,l1+l2+l3)==-1: continue # odd parity
                for m1 in range(-l1,l1+1):
                    for m2 in range(-l2,l2+1):
                        m3 = -m1-m2
                        if abs(m3)>l3: continue
                        weight = WeightFunction(l1,l2,l3,m1,m2,m3)
                        if weight!=0:
                            WeightsNPCF[index] = weight
                        index += 1

    # trim down to remove trivial elements
    WeightsNPCF = WeightsNPCF[:index]

if N==5:
    # 5PCF computation
    def WeightFunction(l1,l2,l12,l3,l4,m1,m2,m3,m4):
        """C_m^Lambda for 4th order isotropic basis function"""
        pref = pow(-1.,l1+l2+l3+l4)*np.sqrt(2.*l12+1.)
        summ = wigner_3j(l1,l2,l12,m1,m2,-m1-m2)*wigner_3j(l12,l3,l4,m1+m2,m3,m4)*pow(-1.,l12-m1-m2)
        return summ*pref

    # Create 1D output matrix. The size is an overestimate and will be trimmed down later.
    WeightsNPCF = np.zeros(((LMAX+1)**8))
    index = 0
    for l1 in range(LMAX+1):
        for l2 in range(LMAX+1):
            for l12 in range(abs(l1-l2),min(LMAX,l1+l2)+1):
                for l3 in range(LMAX+1):
                    for l4 in range(abs(l12-l3),min(LMAX,l12+l3)+1):
                        if pow(-1.,l1+l2+l3+l4)==-1: continue # odd parity
                        for m1 in range(-l1,l1+1):
                            for m2 in range(-l2,l2+1):
                                if abs(m1+m2)>l12: continue # m12 condition
                                for m3 in range(-l3,l3+1):
                                    m4 = -m1-m2-m3
                                    if abs(m4)>l4: continue
                                    weight = WeightFunction(l1,l2,l12,l3,l4,m1,m2,m3,m4)
                                    if weight!=0:
                                        WeightsNPCF[index] = weight
                                    index += 1

    # trim down to remove trivial elements
    WeightsNPCF = WeightsNPCF[:index]

if N==6:
    # 6PCF computation
    def WeightFunction(l1,l2,l12,l3,l123,l4,l5,m1,m2,m3,m4,m5):
        """C_m^Lambda for 5th order isotropic basis function"""
        pref = pow(-1.,l1+l2+l3+l4+l5)*np.sqrt(2.*l12+1.)*np.sqrt(2.*l123+1.)
        # noting m12 = m1+m2, m123 = m1+m2+m3
        summ = wigner_3j(l1,l2,l12,m1,m2,-m1-m2)*wigner_3j(l12,l3,l123,m1+m2,m3,-m1-m2-m3)*wigner_3j(l123,l4,l5,m1+m2+m3,m4,m5)*pow(-1.,l12-m1-m2+l123-m1-m2-m3)
        return summ*pref

    # Create 1D output matrix. The size is an overestimate and will be trimmed down later.
    WeightsNPCF = np.zeros(((LMAX+1)**11))
    index = 0
    for l1 in range(LMAX+1):
        for l2 in range(LMAX+1):
            for l12 in range(abs(l1-l2),min(LMAX,l1+l2)+1):
                for l3 in range(LMAX+1):
                    for l123 in range(abs(l12-l3),min(LMAX,l12+l3)+1):
                        for l4 in range(LMAX+1):
                            for l5 in range(abs(l123-l4),min(LMAX,l123+l4)+1):
                                if pow(-1.,l1+l2+l3+l4+l5)==-1: continue # odd parity
                                for m1 in range(-l1,l1+1):
                                    for m2 in range(-l2,l2+1):
                                        if abs(m1+m2)>l12: continue # m12 condition
                                        for m3 in range(-l3,l3+1):
                                            if abs(m1+m2+m3)>l123: continue # m123 condition
                                            for m4 in range(-l4,l4+1):
                                                m5 = -m1-m2-m3-m4
                                                if abs(m5)>l5: continue
                                                weight = WeightFunction(l1,l2,l12,l3,l123,l4,l5,m1,m2,m3,m4,m5)
                                                if weight!=0:
                                                    WeightsNPCF[index] = weight
                                                index += 1

    # trim down to remove trivial elements
    WeightsNPCF = WeightsNPCF[:index]

outfile = 'coupling_matrices/weights_%dpcf_LMAX%d.txt'%(N,LMAX)
np.savetxt(outfile,WeightsNPCF)
print("Coefficients saved to %s"%outfile)
