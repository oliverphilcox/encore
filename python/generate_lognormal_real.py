### generate_lognormal_real.py (Oliver Philcox, 2021)
# Generate a lognormal galaxy catalog without RSD, using nbodykit.
# This is useful to test the GRF model at high-z (where non-linear corrections are small)
# The script will generate a set of output GRFs in the given output directory.

#### MODULES ####
from nbodykit.lab import *
import numpy as np
import os,sys,time
from mcfit import P2xi, xi2P

#### INPUTS ####
if len(sys.argv)!=3:
    raise Exception("Need to specify minimum and maximum bin!")
else:
    sim_min = int(sys.argv[1])
    sim_max = int(sys.argv[2])

#### PARAMETERS ####
# (vaguely matching CMASS-N)
vol = 3.9e9
Ngal = 587071
nbar = Ngal*1./vol
BoxSize = int(vol**(1./3.))
Nmesh = 512
b1 = 1.9
#LOS = [0,0,1] # line-of-sight
z = 2. # redshift

print("### PARAMETERS ###")
print("# BoxSize: %d"%BoxSize)
print("# N_grid: %d"%Nmesh)
print("# Linear Bias: %.1f"%b1)
print("# Redshift: %.2f"%z)
print("# n_bar: %.2e"%nbar)
print("##################")

outdir = '/projects/QUIJOTE/Oliver/npcf/real_lognormal/mocks/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Cosmology
OmegaM = 0.307115
OmegaL = 0.692885
Omegab = 0.048206
A_s = 2.1467e-9
h = 0.6777
n_s = 0.9611
N_ncdm = 0
N_ur = 3.046

init = time.time()

# Define cosmology
cosmo = cosmology.Cosmology(h=h,A_s=A_s,n_s=n_s,N_ncdm=N_ncdm,N_ur=N_ur,Omega_b=Omegab,Omega_cdm=OmegaM-Omegab)
Plin = cosmology.LinearPower(cosmo, z, transfer='CLASS')
pk_interp = lambda k: Plin(k)

# Compute and save Pk and xi multipoles
k_arr = np.logspace(-5,3,10000)
#fz = cosmo.scale_independent_growth_rate(z)

pk_m = Plin(k_arr)
#pk0 = pk_m*(b1**2.+2./3.*b1*fz+1./5.*fz**2.)
pk0 = pk_m*(b1**2.)
#pk2 = pk_m*(4.*b1*fz/3.+4.*fz**2/7.)
#pk4 = pk_m*8./35.*fz**2.
r_arr,xi0 = P2xi(k_arr,l=0)(pk0)
#xi2 = P2xi(k_arr,l=2)(pk2)[1]
#xi4 = P2xi(k_arr,l=4)(pk4)[1]

np.savetxt(outdir+'input_pk_lognormal_real.txt',np.vstack([k_arr,pk0]).T)#,pk2,pk4]).T)
np.savetxt(outdir+'input_xi_lognormal_real.txt',np.vstack([r_arr,xi0]).T)#,xi2,xi4]).T)
print("Saved model P_ell(k) and xi_ell(r) to %s"%outdir)

N_cat = 0
for i in range(sim_min,sim_max+1):
    print("Creating catalog %d"%(i))

    # Create log-normal catalog
    cat = LogNormalCatalog(Plin=pk_interp,cosmo=cosmo,redshift=z,nbar=nbar, BoxSize=BoxSize, Nmesh=Nmesh, bias=b1, seed=i)
    #cat['RSDPosition'] = (cat['Position']+cat['VelocityOffset']*LOS)%BoxSize
    cat['Weight'] = np.ones(len(cat))*1.

    # Save to file
    #output = np.vstack([cat['RSDPosition'].compute().T,cat['Weight']]).T
    output = np.vstack([cat['Position'].compute().T,cat['Weight']]).T
    np.savetxt(outdir+'lognormal%d.data'%i,output)

    # Also convert to .gz file
    os.system('gzip -f %s'%(outdir+'lognormal%d.data'%i))

    N_cat += 1

print("Created %d catalogs with N_grid = %d after %.2f seconds"%(N_cat,Nmesh,time.time()-init))
