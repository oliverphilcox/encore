import sys, os, time
from scipy.interpolate import interp1d
from mcfit import P2xi, xi2P
from nbodykit.lab import *
from nbodykit import setup_logging, style, cosmology
import numpy as np

from nbodykit.source.catalog import ArrayCatalog
from nbodykit.transform import StackColumns
from nbodykit.mockmaker import gaussian_real_fields, gaussian_complex_fields, poisson_sample_to_points
from nbodykit.cosmology.power.linear import LinearPower
from nbodykit.mpirng import MPIRandomState
from pmesh.pm import RealField, ComplexField, ParticleMesh
import mpsort

import sys,os

## Find sim numbers
if len(sys.argv)!=3:
    raise Exception("Need to specify min / max sim numbers!")

sim_min = int(sys.argv[1])
sim_max = int(sys.argv[2])

Pk_in = np.loadtxt('/projects/QUIJOTE/Oliver/npcf/grf/pk_m_linear_input.txt')

class GenerateGRFMulti(object):

    def __init__(self, Nmesh, Pk_in=None, croot=None,add_RSD=True):
        """Generate a GRF with a given input power spectrum. The power spectrum should specify only linear power (no bias)."""

        alpha = 1 # leave it as 1
        self.nobj = int(587071/alpha)
        self.vol = 3.9e9/alpha
        self.lbox = int(self.vol**(1./3))
        self.nbar = self.nobj/self.vol
        self.Nmesh = Nmesh
        self.add_RSD = add_RSD

        self.bias = 1.9
        self.z = 0.57
        self.growth_rate = 0.7770083924907131

        if croot is not None:
            self.croot = croot # dir and name to save the catalog
        if Pk_in is not None:
            self.Pk_in = Pk_in
        else:
            self.Pk_in = None

        if self.add_RSD:
            self.los = np.asarray([0,0,1])

        print("#obj=", self.nobj, "vol=%.2f [Gpc/h]^3" %(self.vol/1e9),
              "lbox=%.1f [Mpc/h]"% self.lbox, "nbar= %.3e"%self.nbar)

        # Create power spectrum
        if self.Pk_in is None:
            self.init_plin()
        self.init_Pk_interp()

        # Create mesh
        self.pm = ParticleMesh(Nmesh=[self.Nmesh, self.Nmesh, self.Nmesh],
                          BoxSize=self.lbox)

    def init_Pk_interp(self):
        self.k_logspace = numpy.logspace(-3, 1.5, 200)
        if self.Pk_in is None:
            if self.Plin is None:
                print("please init linear Pk at first ...")
            Pk_gal = self.Plin(self.k_logspace)
            self.Pk_gal_interp = interp1d(self.k_logspace, Pk_gal,
                                          kind='cubic', bounds_error=None,
                                          fill_value="extrapolate")
        else:
            kk_in = self.Pk_in[:,0]
            Pk_gal = self.Pk_in[:,1]
            self.Pk_gal_interp = interp1d(kk_in, Pk_gal,
                                          kind='cubic', bounds_error=None,
                                          fill_value="extrapolate")
    def init_plin(self):
        h = 0.676
        Omega_nu = 0.00140971
        Omega0_m = 0.31
        Omega0_b = 0.022/h**2
        Omega0_cdm = Omega0_m - Omega0_b - Omega_nu
        n_s = 0.96
        sigma8 = 0.824

        cosmo = cosmology.Cosmology(h=h, Omega0_b=Omega0_b,
                                    Omega0_cdm=Omega0_cdm, n_s=n_s)
        cosmo.match(sigma8=sigma8)
        redshift = 0.57
        self.Plin = LinearPower(cosmo, redshift, transfer='CLASS')

    def mk_catalog_grf_weighted(self,seed):

        rand_cat = RandomCatalog(self.nobj)
        rand_cat['Position'] = rand_cat.rng.uniform(itemshape=(3,))*self.lbox

        if self.add_RSD:
            ### Add Kaiser RSD by applying LoS-gradient via Fourier space
            self.idx_los = numpy.where(self.los==1)[0][0]
            
            # Start by generating GRF in complex space
            # Linear power spectrum does *not* include bias
            self.deltak = gaussian_complex_fields(pm=self.pm,
                                linear_power=self.Pk_gal_interp,
                                compute_displacement=False, seed=seed,
                                inverted_phase=False, unitary_amplitude=False)[0]
            # Compute k-hat
            kgrid = [kk.astype('f8') for kk in self.deltak.slabs.optx]
            knorm = numpy.sqrt(sum(kk**2 for kk in kgrid))
            knorm[knorm==0.] = numpy.inf
            kgrid = [k/knorm for k in kgrid]
            # Compute k-hat dotted with LoS
            k_los = kgrid[self.idx_los]
            # Now transform to redshift-space
            field_k = self.deltak.copy()
            field_k.value *= (self.bias + self.growth_rate*k_los**2)
            field_rsd = field_k.c2r()
            #field_rsd.value += 1.
            field_3col = field_rsd.readout(rand_cat['Position'].compute(),
                                                     resampler='nnb')
            rand_cat['Weight'] = field_3col#/(numpy.sum(self.field_3col)/self.nobj)-(1-self.mean_weight)
        else:
            ## No RSD here. Generate the GRF in real-space
            # Linear power spectrum includes bias here.
            deltar = gaussian_real_fields(pm=self.pm,
                                linear_power=lambda k: self.bias**2*self.Pk_gal_interp(k),
                                compute_displacement=False, seed=seed,
                                inverted_phase=False, unitary_amplitude=False)[0]
            field = deltar.copy()# + 1.

            field_3col = field.readout(rand_cat['Position'].compute(),
                                                      resampler='nnb')
            rand_cat['Weight'] = field_3col
        self.catalog = rand_cat.copy()

    def save_catalog_ascii(self,index):
        print("Saving mock with index %d"%index)
        nrows, ncols = self.catalog.csize, int(len(self.catalog.columns)-2+2)
        cat_ascii = numpy.zeros([nrows, ncols])
        cat_ascii[:,:3] = self.catalog['Position'][:,:3].compute()
        cat_ascii[:,3]  = self.catalog['Weight'].compute()

        if self.add_RSD:
            numpy.savetxt(self.croot+'rsd_grf%d.txt'%index, cat_ascii)
        else:
            numpy.savetxt(self.croot+'grf%d.txt'%index, cat_ascii)

    def run_mk_catalog_grf_weighted(self, index, do_save_catalog):

        seed = numpy.random.randint(0,0xfffffff)
        self.mk_catalog_grf_weighted(seed)
        if do_save_catalog:
            if self.croot is None:
                print("Please specify where to save catalog ...")
            self.save_catalog_ascii(index)

### NOW SAVE SIMS
print("Loading GRF field")
grf = GenerateGRFMulti(512,Pk_in=Pk_in,croot='/projects/QUIJOTE/Oliver/npcf/rsd_grf/mocks/',add_RSD=True)
for i in range(sim_min,sim_max+1):
    grf.run_mk_catalog_grf_weighted(i,1)

print("Finished generating sims %d to %d"%(sim_min,sim_max))
