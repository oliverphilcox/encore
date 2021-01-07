### combine_files_periodic.py (Oliver Philcox, 2021)
# This reads in a set of (data-random) and (random) particle counts and uses them to construct the N-point functions, assuming a periodic geometry
# It is designed to be used with the run_npcf.csh script
# Currently 3PCF, 4PCF, 5PCF and 6PCF are supported.
# The output is saved to the working directory with the same format as the NPCF counts, with the filename ...zeta_{N}pcf.txt

import sys, os
import numpy as np

## First read-in the input file string from the command line
if len(sys.argv)!=2:
    raise Exception("Need to specify the input files!")
else:
    inputs = str(sys.argv[1])

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

    # Now compute full NPCF
    # We do not apply any edge-correction, such that zeta = (D-R)_Lambda / R_{Lambda=0}, i.e. we assume R_{Lambda} = 0 for Lambda > (00...0)

    assert ell_1[0]==0
    zeta = np.zeros_like(countsN)
    for i in range(len(bin1)):
        zeta[:,i] = countsN[:,i]/countsR[0,i]

    # Now save the output to file, copying the first few lines from the N files
    zeta_file = inputs+'.zeta_%dpcf.txt'%N
    rfile = open(R_file,"r")
    zfile = open(zeta_file,"w")
    for l,line in enumerate(rfile):
        if l>=4+N: continue
        zfile.write(line)
    for a in range(len(zeta)):
        zfile.write("%d\t"%ell_1[a])
        for b in range(len(zeta[a])):
            zfile.write("%.8e\t"%zeta[a,b])
        zfile.write("\n")
    zfile.close()

    print("Computed %dPCF using %d (data-random) files, saving to %s\n"%(N,N_files,zeta_file))
