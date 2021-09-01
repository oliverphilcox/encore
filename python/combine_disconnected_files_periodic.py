### combine_disconnected_files_periodic.py (Oliver Philcox, 2021)
# This reads in a set of (data-random) and (random) particle counts and uses them to construct the disconnected N-point functions, assuming a periodic geometry.
# It is designed to be used with the run_npcf.csh script
# Currently only the 4PCF is supported.
# The output is saved to the working directory with the same format as the NPCF counts, with the filename ...zeta_discon_{N}pcf.txt

import sys, os, time
import subprocess
import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j

## First read-in the input file string from the command line
if len(sys.argv)!=7:
    raise Exception("Need to specify the input files, N, number of galaxies, boxsize, R_min and R_max!")
else:
    inputs = str(sys.argv[1])
    N = int(sys.argv[2])
    N_gal = int(sys.argv[3])
    boxsize = np.float64(sys.argv[4])
    R_min = np.float64(sys.argv[5])
    R_max = np.float64(sys.argv[6])
n_bar = np.float64(N_gal)/boxsize**3.

print("Reading in files starting with %s\n"%inputs)
init = time.time()

if N!=4:
    raise Exception("Only N=4 is implemented!")

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

#################### COMPUTE XI_LM ####################

#### Load in first D-R piece to get bins
tmp_file = inputs+'.n00_2pcf_mult1.txt'
counts_tmp = np.loadtxt(tmp_file,skiprows=7) # skipping rows with radial bins
l1_lm, m1_lm = np.asarray(counts_tmp[:,:2],dtype=int).T
bin1_lm = np.asarray(np.loadtxt(tmp_file,skiprows=6,max_rows=1),dtype=int)

n_ell = len(np.unique(l1_lm))
lmax = n_ell-1
n_r = len(bin1_lm)
nlm = n_ell**2

#### Load in NN_lm piece
countsN_all = []
total_DmR = 0
for i in range(100):
    DmR_file = inputs+'.n%s_2pcf_mult1.txt'%(str(i).zfill(2))
    if not os.path.exists(DmR_file): continue
    # Extract counts
    tmp_counts = np.loadtxt(DmR_file,skiprows=7)
    # Read-in and take complex conjugate (since definition involves a_lm^*)
    countsN_all.append(tmp_counts[:,2::2]-1.0j*tmp_counts[:,3::2])
countsN_all = np.asarray(countsN_all)
N_files = len(countsN_all)
NN_lm = np.mean(countsN_all,axis=0)

# Function to compute radius from bin index
radius = lambda bin, n_r: R_min + 1.*bin*(R_max-R_min)/n_r
# Function to compute bin volume * n-bar from bin index
nV = lambda bin, n_r: n_bar*4.*np.pi/3.*(radius(bin+1,n_r)**3.-radius(bin,n_r)**3.)

### Now compute analytic randoms for ell=m=0
# Note that we must include the basis function Y_{00} = 1/sqrt{4 pi}
n_r_lm = 1.*len(np.unique(bin1_lm))
RR_analyt = N_gal*nV(bin1_lm,n_r)/np.sqrt(4.*np.pi)

#### Compute xi_lm, including factor of sqrt{4 pi} from Y_00
# We do not apply any edge-correction, such that zeta = NN_{lm} / RR_{00} * (4 pi)^{1/2}
# i.e. we assume R_{00} = 0 for (lm) > (00)
# the (4 pi) factor comes from the Y_{00} basis function.
xi_lm = NN_lm/RR_analyt*np.sqrt(4.*np.pi)

print("Computed periodic xi_lm multipoles after %.2f seconds"%(time.time()-init))

#################### COMPUTE XI_{LML'M'} ####################

#### Load in first RNN_{lml'm'} piece to get bins
tmp_file = inputs+'.n00_2pcf_mult2.txt'
counts_tmp = np.loadtxt(tmp_file,skiprows=8) # skipping rows with radial bins
l1_lmlm, m1_lmlm, l2_lmlm, m2_lmlm = np.asarray(counts_tmp[:,:4],dtype=int).T
bin1_lmlm, bin2_lmlm = np.asarray(np.loadtxt(tmp_file,skiprows=6,max_rows=2),dtype=int)

assert len(np.unique(l1_lmlm))==n_ell
assert len(np.unique(bin1_lm))==n_r

#### Load in RNN_{lml'm'} piece
countsN_all = []
total_DmR = 0
for i in range(100):
    DmR_file = inputs+'.n%s_2pcf_mult2.txt'%(str(i).zfill(2))
    if not os.path.exists(DmR_file): continue
    # Extract counts
    tmp_counts = np.loadtxt(DmR_file,skiprows=8)
    # Read-in and take complex conjugate (since definition involves a_lm^*)
    countsN_all.append(tmp_counts[:,4::2]-1.0j*tmp_counts[:,5::2])
countsN_all = np.asarray(countsN_all)
assert len(countsN_all)==N_files
RNN_lmlm = np.mean(countsN_all,axis=0)*-1. # add -1 due to weight inversion

# Compute collapsed indices
index1 = l1_lmlm**2+l1_lmlm+m1_lmlm
index2 = l2_lmlm**2+l2_lmlm+m2_lmlm

### Now compute analytic randoms for ell=m=ell'=m'=0
# Note that we must include the basis function Y_{00}Y_{00} = 1/(4 pi)
n_r_lm = 1.*len(np.unique(bin1_lmlm))
RRR_analyt = N_gal*nV(bin1_lmlm,n_r)*nV(bin2_lmlm,n_r)/(4.*np.pi)

#### Compute xi_lml'm', including factor of 4 pi from Y_00 Y_00
# We do not apply any edge-correction, such that zeta = (RNN)_{lml'm'} / RRR_{0000} * (4 pi)
# i.e. we assume R_{lml'm'} = 0 for (lml'm') > (0000)
# the (4 pi) factor comes from the Y_{00}Y_{00} basis function.
xi_lmlm = RNN_lmlm/RRR_analyt*4.*np.pi

print("Computed periodic xi_{lml'm'} multipoles after %.2f seconds"%(time.time()-init))

#################### COMBINE TO OBTAIN 4PCF ####################

#### Compute 4PCF coupling matrix if necessary
coupling_file = get_script_path()+'/../coupling_matrices/disconnected_4pcf_coupling_lmax%d.npy'%lmax
try:
    fourpcf_coupling = np.load(coupling_file)
    print("Loading 4PCF coupling matrix from file")
except IOError:
    print("Computing 4PCF coupling matrix for l_max = %d"%lmax)

    # Form 4PCF coupling matrix, i.e. ThreeJ[l1,l2,l3,m1,m2,m3]
    fourpcf_coupling = np.zeros((nlm,nlm,nlm))
    for l in range(n_ell):
        for m in range(-l,l+1):
            for lp in range(n_ell):
                for mp in range(-lp,lp+1):
                    for L in range(n_ell):
                        for M in range(-L,L+1):
                            tj = wigner_3j(l,lp,L,m,mp,M)
                            if tj==0: continue
                            fourpcf_coupling[l**2+l+m,lp**2+lp+mp,L**2+L+M] = tj

    # Save matrix to file
    np.save(coupling_file,fourpcf_coupling)
    print("\nSaved 4PCF coupling matrix to %s"%coupling_file)

### Define an output matrix shape and arrays
ct_ell = 0
ell_1, ell_2, ell_3 = [],[],[]
for l1 in range(lmax+1):
    for l2 in range(lmax+1):
        for l3 in range(lmax+1):
            if pow(-1.,l1+l2+l3)==-1: continue
            if l3<np.abs(l1-l2): continue
            if l3>l1+l2: continue
            ct_ell+=1
            ell_1.append(l1)
            ell_2.append(l2)
            ell_3.append(l3)

ct_r = 0
bin1, bin2, bin3 = [],[],[]
for b1 in range(n_r):
    for b2 in range(b1+1,n_r):
        for b3 in range(b2+1,n_r):
            ct_r += 1
            bin1.append(b1)
            bin2.append(b2)
            bin3.append(b3)

zeta_discon = np.zeros((ct_ell,ct_r))

# Sum to accumulate 4PCF
print("Accumulating disconnected %dPCF"%N)
for r_index in range(ct_r):
    b1,b2,b3 = bin1[r_index],bin2[r_index],bin3[r_index]

    ## First permutation

    # Find relevant radial indices
    xi1 = xi_lm[:,bin1_lm==b1][:,0]
    xi2 = xi_lmlm[:,np.logical_and(bin1_lmlm==b2,bin2_lmlm==b3)][:,0]

    # Find angular bins and add to sum
    for l_index in range(ct_ell):
        ell1,ell2,ell3 = ell_1[l_index],ell_2[l_index],ell_3[l_index]

        this_xi1 = xi1[l1_lm==ell1]
        this_xi2 = xi2[np.logical_and(l1_lmlm==ell2,l2_lmlm==ell3)]
        this_coupling = fourpcf_coupling[l1_lm==ell1][:,index1[np.logical_and(l1_lmlm==ell2,l2_lmlm==ell3)],index2[np.logical_and(l1_lmlm==ell2,l2_lmlm==ell3)]]

        zeta_discon[l_index,r_index] += np.real_if_close(np.sum(this_xi1[:,np.newaxis]*this_xi2[np.newaxis,:]*this_coupling))

    ## Second permutation

    # Find angular bins and add to sum
    xi1 = xi_lm[:,bin1_lm==b2][:,0]
    xi2 = xi_lmlm[:,np.logical_and(bin1_lmlm==b1,bin2_lmlm==b3)][:,0]

    for l_index in range(ct_ell):
        ell1,ell2,ell3 = ell_1[l_index],ell_2[l_index],ell_3[l_index]

        this_xi1 = xi1[l1_lm==ell2]
        this_xi2 = xi2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell3)]
        this_coupling = fourpcf_coupling[l1_lm==ell2][:,index1[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell3)],index2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell3)]]

        zeta_discon[l_index,r_index] += np.real_if_close(np.sum(this_xi1[:,np.newaxis]*this_xi2[np.newaxis,:]*this_coupling))

    ## Third permutation

    # Find angular bins and add to sum
    xi1 = xi_lm[:,bin1_lm==b3][:,0]
    xi2 = xi_lmlm[:,np.logical_and(bin1_lmlm==b1,bin2_lmlm==b2)][:,0]

    for l_index in range(ct_ell):
        ell1,ell2,ell3 = ell_1[l_index],ell_2[l_index],ell_3[l_index]

        this_xi1 = xi1[l1_lm==ell3]
        this_xi2 = xi2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell2)]
        this_coupling = fourpcf_coupling[l1_lm==ell3][:,index1[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell2)],index2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell2)]]

        zeta_discon[l_index,r_index] += np.real_if_close(np.sum(this_xi1[:,np.newaxis]*this_xi2[np.newaxis,:]*this_coupling))

# Now save the output to file, copying the first few lines from the N files
zeta_file = inputs+'.zeta_discon_%dpcf.txt'%N
R_file = inputs+'.r_3pcf.txt'
rfile = open(R_file,"r")
zfile = open(zeta_file,"w")
for l,line in enumerate(rfile):
    if l>=4: continue
    zfile.write(line)
zfile.write("## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Rows 4+ = zeta^{(disc)}_l1l2l3^abc\n");
zfile.write("## Columns 1-3 specify the (l1, l2, l3) multipole triplet\n");
zfile.write("\t\t\t")
for i in range(ct_r): zfile.write("%d\t"%bin1[i])
zfile.write("\n")
zfile.write("\t\t\t")
for i in range(ct_r): zfile.write("%d\t"%bin2[i])
zfile.write("\n")
zfile.write("\t\t\t")
for i in range(ct_r): zfile.write("%d\t"%bin3[i])
zfile.write("\n")

for a in range(ct_ell):
    zfile.write("%d\t"%ell_1[a])
    zfile.write("%d\t"%ell_2[a])
    zfile.write("%d\t"%ell_3[a])
    for b in range(ct_r):
        zfile.write("%.8e\t"%zeta_discon[a,b])
    zfile.write("\n")
zfile.close()

print("Disconnected %dPCF saved to %s"%(N,zeta_file))
print("Exiting after %.2f seconds"%(time.time()-init))
sys.exit();
