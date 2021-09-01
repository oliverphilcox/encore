from nbodykit.lab import *
from nbodykit.cosmology.correlation import pk_to_xi, xi_to_pk
import mcfit, pickle
from mcfit import P2xi, xi2P

import itertools
from itertools import combinations, product, permutations
import dask.array as da
from scipy.special import sici
from scipy.integrate import quad, cumtrapz
from scipy.special import spherical_jn
from scipy.interpolate import interp1d
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j



class model_cov_4PCF(object):

    def __init__(self, meta_data=None, fname_f3l=None, fname_GLs=None, do_rsd=False, vol=3.9e+09, bias=2.2, k_in=None, Pk_in=None,
                 shotnoise=None, r_in=None, xi_in=None, verbose=False):

        '''
            meta_data: settings about rbins, ells
            fname_f3l: file to load f3l tensor.
            fname_GLs: file to load coefficients
                kernels and precomputed coefficients can be found here:
                fdir = "/Users/mianoctobers/Projects/NPCF/results/kernels_4pcf_cov_model/"
                "flll_iso_4pcf.pkl": rbin = [min=8.5, max=170, step=17]
                "flll_iso_4pcf_logn_real_27x153.pkl": rbin = [min=27, max=153, step=14]
                "coeffs_LG_4pcf_block1.pkl": rbin = [min=8.5, max=170, step=17]
                "coeffs_LG_4pcf_block1_27x153.pkl": rbin = [min=27, max=153, step=14]
        '''

        self.meta_data = meta_data
        self.fname_f3l = fname_f3l
        self.fname_GLs = fname_GLs
        self.verbose = verbose
        self.do_rsd = do_rsd
        self.vol = vol
        self.bias = bias

        if Pk_in is not None:
            self.k_in = k_in
            self.Pk_in = Pk_in
        else:
            self.Pk_in = None

        if shotnoise is not None:
            self.shot_noise = shotnoise
            print("Why are we multiplying Pk by this??")
            self.Pk_in = self.Pk_in * numpy.exp(-k_in**1) + shotnoise

        if r_in is not None:
            self.r_in = r_in
        if xi_in is not None:
            self.xi_in = xi_in
        else:
            self.xi_in = None


    def run(self):

        self.init_arrs()
        self.init_2stat()
        self.gen_jnbar_dict()
        if self.fname_f3l is not None:
            fdir = "/Users/mianoctobers/Projects/NPCF/results/kernels_4pcf_cov_model"
            self.io_flll(action='load', fname=fdir+self.fname_f3l)
        else:
            print("f3l tensor has not been pre-computed yet, do you want to calculate now?")
        if self.fname_GLs is not None:
            fdir = "/Users/mianoctobers/Projects/NPCF/results/kernels_4pcf_cov_model/"
            self.io_GLs(action='load', fname=fdir+self.fname_GLs)
        self.get_4pcf_cov_model()

    def init_arrs(self):

        if self.r_in is None:
            sbin_min= 8.5
            sbin_max = 170
            dsbin = 17
            self.rr= numpy.arange(sbin_min, sbin_max, dsbin)
        else:
            self.rr= self.r_in
        self.nbins = len(self.rr)
        kbin_min= 1e-4
        kbin_max= 5.
        nbink = 5000
        self.kk = numpy.linspace(kbin_min, kbin_max, nbink)
        self.kk_log = numpy.logspace(numpy.log10(kbin_min), numpy.log10(kbin_max), nbink)
        self.dkk = self.kk[1]-self.kk[0]
        self.rrv, self.kkv = numpy.meshgrid(self.rr, self.kk)

    def init_cosmo(self):
        h = 0.676
        Omega_nu = 0.00140971
        Omega0_m = 0.31
        Omega0_b = 0.022/h**2
        Omega0_cdm = Omega0_m - Omega0_b - Omega_nu
        n_s = 0.96
        sigma8 = 0.824
        self.cosmo = cosmology.Cosmology(h=h, Omega0_b=Omega0_b,
                                    Omega0_cdm=Omega0_cdm, n_s=n_s)
        self.cosmo.match(sigma8=sigma8)

    def init_2stat(self):

        if self.Pk_in is not None:
            pk_interp = interp1d(self.k_in, self.Pk_in, kind='cubic', bounds_error=False, fill_value=0)
            self.Pk = pk_interp(self.kk)
            if self.xi_in is not None:
                self.r = self.r_in.copy()
                self.xi_interp = interp1d(self.r_in, self.xi_in, kind='cubic', fill_value='extrapolate')
            else:
                Pk_log = pk_interp(self.kk_log)
                self.r, self.xi = P2xi(self.kk_log)(Pk_log)
                self.xi_interp = interp1d(self.r, self.xi, kind='cubic', fill_value='extrapolate')

        elif self.Pk_in is None:
            self.init_cosmo()
            Plin = cosmology.LinearPower(self.cosmo, redshift=0.57, transfer='CLASS')
            self.Pk = Plin(self.kk)
            Pk_log = Plin(self.kk_log)
            self.r, self.xi = P2xi(self.kk_log)(Pk_log)
            self.xi_interp = interp1d(self.r, self.xi, kind='cubic', fill_value='extrapolate')
        if self.do_rsd:
            if not hasattr(self, 'cosmo'):
                print("init cosmo ..")
                self.init_cosmo()
            self.get_xibar()
            growth_rate = self.cosmo.scale_independent_growth_rate(0.57)
            beta = growth_rate/self.bias
            self.Pk_ell = {}
            self.xi_ell_interp = {}
            self.Pk_ell[0] = self.Pk * (1 + 2*beta/3 + beta**2/5) * self.bias**2
            xi0 = self.xi_interp(self.r) * (1 + 2*beta/3 + beta**2/5) * self.bias**2
            self.xi_ell_interp[0] = interp1d(self.r, xi0, kind='cubic', fill_value='extrapolate')
            self.Pk_ell[1] = 0
            self.xi_ell_interp[1] = interp1d(self.r, numpy.zeros_like(self.r), kind='cubic', fill_value='extrapolate')
            self.Pk_ell[2] = self.Pk * (4*beta/3 + 4*beta**2/7) * self.bias**2
            xi2 = (self.xi_interp(self.r)-self.xi_bar) * (4*beta/3 + 4*beta**2/7) * self.bias**2
            self.xi_ell_interp[2] = interp1d(self.r, xi2, kind='cubic', fill_value='extrapolate')
            self.Pk_ell[3] = 0
            self.xi_ell_interp[3] = interp1d(self.r, numpy.zeros_like(self.r), kind='cubic', fill_value='extrapolate')
            self.Pk_ell[4] = self.Pk * (8*beta**2/35) * self.bias**2
            xi4 = (self.xi_interp(self.r)-self.xi_bar_bar) * (8*beta**2/35) * self.bias**2
            self.xi_ell_interp[4] = interp1d(self.r, xi4, kind='cubic', fill_value='extrapolate')
            self.Pk_ell[5] = 0
            self.xi_ell_interp[5] = interp1d(self.r, numpy.zeros_like(self.r), kind='cubic', fill_value='extrapolate')

    def get_xibar(self):
        ss = numpy.linspace(1e-2, 200, 1e3)
        ds = numpy.average(ss[1:]-ss[:-1])
        self.xi_bar = numpy.zeros(len(self.r))
        self.xi_bar_bar = numpy.zeros(len(self.r))
        for ii in range(len(self.r)):
            si = ss[ss < self.r[ii]]
            self.xi_bar[ii] = numpy.sum(self.xi_interp(si)*ds*si**2)/self.r[ii]**3*3
            self.xi_bar_bar[ii] = numpy.sum(self.xi_interp(si)*ds*si**4)/self.r[ii]**5*5

    def get_ells(self, verbose=False):

        '''
            calculate all ells terms needed
            for covariance matrix
            Cov_ll'(R, R')

        '''

        from itertools import combinations

        self.ells_comb2 = list(combinations(self.meta_data.ells, 2))
        self.ells_comb2 = [(ii, ii) for ii in self.meta_data.ells] + self.ells_comb2
        self.ells_g_comb2 = list(combinations(self.meta_data.ell_gaussian, 2))
        self.ells_g_comb2 = [(ii, ii) for ii in self.meta_data.ell_gaussian] + self.ells_g_comb2
        self.ells_ng_comb2 = list(combinations(self.meta_data.ell_non_gaussian, 2))
        self.ells_ng_comb2 = [(ii, ii) for ii in self.meta_data.ell_non_gaussian] + self.ells_ng_comb2

        if verbose:
            print("get all ells, done!")

    def get_lambda_pp_noperm(self, lls, lls_p):

        '''
            calculate ell'' according to
            ell and ell' for triangulation:
            1. ell'' >= abs(ell - ell')
            2. ell'' <= ell + ell'

        '''

        l1, l2, l3 = lls
        l1_p, l2_p, l3_p = lls_p

        l1_pp_l = []
        l2_pp_l = []
        l3_pp_l = []

        for l1_pp in range(abs(l1-l1_p), l1+l1_p+1, 2):
            l1_pp_l.append(l1_pp)
        for l2_pp in range(abs(l2-l2_p), l2+l2_p+1, 2):
            l2_pp_l.append(l2_pp)
        for l3_pp in range(abs(l3-l3_p), l3+l3_p+1, 2):
            l3_pp_l.append(l3_pp)

        ll_pp = [l1_pp_l, l2_pp_l, l3_pp_l]
        ll_pp = list(product(*ll_pp))

        lls_pp = []
        for iell in ll_pp:
            if numpy.mod(iell[0]+iell[1]+iell[2], 2) == 0:
                if (iell[2] >= abs(iell[0]-iell[1])) and (iell[2] <= (iell[0]+iell[1])):
                    lls_pp.append(list(iell))

        return lls_pp

    def permute_like_bob(self, r_in, idx_six):

        '''
            wrap angular momenta orders for fully coupled pieces
            rs follow the following permutation:

                r1 r2 r3
                r2 r1 r3
                r1 r3 r2
                r3 r2 r1
                r3 r1 r2
                r2 r3 r1
        '''

        r1, r2, r3 = r_in

        if idx_six == 1:
            r1, r2, r3 = r2, r1, r3
        elif idx_six == 2:
            r1, r2, r3 = r1, r3, r2
        elif idx_six == 3:
            r1, r2, r3 = r3, r2, r1
        elif idx_six == 4:
            r1, r2, r3 = r3, r1, r2
        elif idx_six == 5:
            r1, r2, r3 = r2, r3, r1

        return [r1, r2, r3]

    def permute_even(self, ells, which_group):

        '''
            123
            231
            312
        '''

        l1, l2, l3 = ells

        if which_group == "block1_case3":
            l1, l2, l3 = l2, l3, l1
        elif which_group == "block1_case4":
            l1, l2, l3 = l3, l1, l2

        return [l1, l2, l3]

    def permute_ell_GH(self, lls, lls_p, which_group, idx_six):

        '''
            ells (l1, l2, l3) as l_Gs like:
            123
            213
            132
            321
            312
            231
        '''

        l_Gs = self.permute_like_bob(lls, idx_six)
        l_Hs = self.permute_even(lls_p, which_group)

        return l_Gs, l_Hs

    def get_all_possible_lll(self):

        '''
            find all possible combinations
            for given triangular rules.
            The output self.ell123
            will be used as input for
            f_lll tensor
        '''

        self.ell123 = []
        groups = ['block1_case1','block1_case2','block1_case3','block1_case4']
        ic = 0
        for l1 in range(0, 5):
            for l2 in range(0, 5):
                for l3 in range(0, 5):
                    for l1_p in range(0, 5):
                        for l2_p in range(0, 5):
                            for l3_p in range(0, 5):
                                l1_max = l1+l1_p+1
                                l2_max = l2+l2_p+1
                                l3_max = l3+l3_p+1
                                l1_min = abs(l1-l1_p)
                                l2_min = abs(l2-l2_p)
                                l3_min = abs(l3-l3_p)
                                for l1_pp in range(l1_min, l1_max, 2):
                                    for l2_pp in range(l2_min, l2_max, 2):
                                        for l3_pp in range(l3_min, l3_max, 2):
                                            ic +=1
                                            if ic % 7e5==0: print(ic)
                                            l1s = [l1, l1_p, l1_pp]
                                            l2s = [l2, l2_p, l2_pp]
                                            l3s = [l3, l3_p, l3_pp]
                                            ell_in = [l1s] + [l2s] + [l3s]
                                            if numpy.mod(l1_pp+l2_pp+l3_pp, 2) == 0:
                                                for ell in ell_in:
                                                    if ell not in self.ell123:
                                                        self.ell123.append(ell)

    def get_spherical_jn(self, ell, is_separation_vec):

        '''
            three vector systems:
                "s": separation vector between two vertices r and r_prime
                "r0": vertices of r0=0 and r0'=0
                "ri": directional vectors
            if is_separation_vec:
                not average jn_bessel
            else:
                average over the bins
        '''

        k = da.array(self.kk)
        self.num_s = 4100
        self.s = da.linspace(1e-5, 1000, self.num_s)
        self.ds = self.s[1] - self.s[0]
        self.ss = self.s.compute()
        self.dss = self.ds.compute() # convert to numpy
        if not is_separation_vec:
            dr = self.rr[1] - self.rr[0]
            r_min = da.from_array(self.rr - dr/2.)
            r_max = da.from_array(self.rr + dr/2)
            rv_min, kv = da.meshgrid(r_min, k, indexing='ij')
            rv_max, kv = da.meshgrid(r_max, k, indexing='ij')
            if ell==0:
                tmp1 = (-kv*rv_max*da.cos(kv*rv_max)+da.sin(kv*rv_max))/kv**3.
                tmp2 = (-kv*rv_min*da.cos(kv*rv_min)+da.sin(kv*rv_min))/kv**3.
            elif ell==1:
                tmp1 = -2.*da.cos(kv*rv_max)/kv**3. - rv_max*da.sin(kv*rv_max)/kv**2.
                tmp2 = -2.*da.cos(kv*rv_min)/kv**3. - rv_min*da.sin(kv*rv_min)/kv**2.
            elif ell==2:
                tmp1 = (rv_max*da.cos(kv*rv_max))/kv**2 - (4*da.sin(kv*rv_max))/kv**3 + (3*sici((kv*rv_max).compute())[0])/kv**3
                tmp2 = (rv_min*da.cos(kv*rv_min))/kv**2 - (4*da.sin(kv*rv_min))/kv**3 + (3*sici((kv*rv_min).compute())[0])/kv**3
            elif ell==3:
                tmp1 = (7.*da.cos(kv*rv_max))/kv**3 - (15.*da.sin(kv*rv_max))/(kv**4*rv_max) + (rv_max*da.sin(kv*rv_max))/kv**2
                # avoid zero errors!
                if rv_min[0,0].compute()==0:
                    tmp2 = -(8./kv**3.)
                else:
                    tmp2 = (7.*da.cos(kv*rv_min))/kv**3 - (15.*da.sin(kv*rv_min))/(kv**4*rv_min) + (rv_min*da.sin(kv*rv_min))/kv**2
            elif ell==4:
                tmp1 = (105.*da.cos(kv*rv_max))/(2.*kv**4*rv_max) - (rv_max*da.cos(kv*rv_max))/kv**2 + (11*da.sin(kv*rv_max))/kv**3 -(105*da.sin(kv*rv_max))/(2.*kv**5*rv_max**2) + (15*sici((kv*rv_max).compute())[0])/(2.*kv**3)
                if rv_min[0,0].compute()==0:
                    tmp2 = 0.
                else:
                    tmp2 = (105.*da.cos(kv*rv_min))/(2.*kv**4*rv_min) - (rv_min*da.cos(kv*rv_min))/kv**2 + (11*da.sin(kv*rv_min))/kv**3 -(105*da.sin(kv*rv_min))/(2.*kv**5*rv_min**2) + (15*sici((kv*rv_min).compute())[0])/(2.*kv**3)
            elif ell==5:
                tmp1 = ((315*kv*rv_max - 16*kv**3*rv_max**3)*da.cos(kv*rv_max) - (315 - 105*kv**2*rv_max**2 + kv**4*rv_max**4)*da.sin(kv*rv_max))/(kv**6*rv_max**3)
                if rv_min[0,0].compute()==0:
                    tmp2 = -16./kv**3.
                else:
                    tmp2 = ((315*kv*rv_min - 16*kv**3*rv_min**3)*da.cos(kv*rv_min) - (315 - 105*kv**2*rv_min**2 + kv**4*rv_min**4)*da.sin(kv*rv_min))/(kv**6*rv_min**3)
            elif ell==6:
                tmp1 = (105*kv**4*rv_max**4*sici((kv*rv_max).compute())[0] + (-176*kv**4*rv_max**4+8505*kv**2*rv_max**2-20790)*da.sin(kv*rv_max)
                        +kv*rv_max*(8*kv**4*rv_max**4-1575*kv**2*rv_max**2+20790)*da.cos(kv*rv_max))/(8*kv**7*rv_max**4)
                if rv_min[0,0].compute()==0:
                    tmp2 = 0
                else:
                    tmp2 = (105*kv**4*rv_min**4*sici((kv*rv_min).compute())[0] + (-176*kv**4*rv_min**4+8505*kv**2*rv_min**2-20790)*da.sin(kv*rv_min)
                            +kv*rv_min*(8*kv**4*rv_min**4-1575*kv**2*rv_min**2+20790)*da.cos(kv*rv_min))/(8*kv**7*rv_min**4)
            elif ell==7:
                tmp1 = (kv*rv_max*(29*kv**4*rv_max**4-2772*kv**2*rv_max**2+27027)*da.cos(kv*rv_max) +
                        (kv**6*rv_max**6-378*kv**4*rv_max**4+11781*kv**2*rv_max**2-27027)*da.sin(kv*rv_max))/(kv**8*rv_max**5)
                if rv_min[0,0].compute()==0:
                    tmp2 = -128/(5*kv**3)
                else:
                    tmp2 = (kv*rv_min*(29*kv**4*rv_min**4-2772*kv**2*rv_min**2+27027)*da.cos(kv*rv_min) +
                            (kv**6*rv_min**6-378*kv**4*rv_min**4+11781*kv**2*rv_min**2-27027)*da.sin(kv*rv_min))/(kv**8*rv_min**5)
            elif ell==8:
                tmp1 = (315*kv**6*rv_max**6*sici((kv*rv_max).compute())[0] + (-16*kv**7*rv_max**7+10395*kv**5*rv_max**5-630630*kv**3*rv_max**3+5405400*kv*rv_max)*da.cos(kv*rv_max)
                        + (592*kv**6*rv_max**6-100485*kv**4*rv_max**4+2432430*kv**2*rv_max**2-5405400))/(16*kv**9*rv_max**6)
                if rv_min[0,0].compute()==0:
                    tmp2 = 0
                else:
                    tmp2 = (315*kv**6*rv_min**6*sici((kv*rv_min).compute())[0] + (-16*kv**7*rv_min**7+10395*kv**5*rv_min**5-630630*kv**3*rv_min**3+5405400*kv*rv_min)*da.cos(kv*rv_min)
                        + (592*kv**6*rv_min**6-100485*kv**4*rv_min**4+2432430*kv**2*rv_min**2-5405400))/(16*kv**9*rv_min**6)

            else:
                raise Exception("not implemented yet!")

            ans = (tmp1-tmp2)/((rv_max**3.-rv_min**3.)/3.)

        else:
            sv, kv = da.meshgrid(self.s, k, indexing='ij')
            ans = spherical_jn(ell, sv*kv)

        return ans

    def gen_jnbar_dict(self, verbose=False):

        '''
            a wrapper for calculating the
            (bin-averaged) sBFs
        '''

        self.jn_bar_r = {}
        self.jn_s = {}
        if verbose:
            print("init jn_bar ..")
        for il in range(0,9):
            if verbose:
                print("*"*il+"%.f"%(il/9*100)+"%", end="\r")
            self.jn_bar_r[il] = self.get_spherical_jn(il, is_separation_vec=False).compute()
            self.jn_s[il] = self.get_spherical_jn(il, is_separation_vec=True).compute()
        if verbose:
            print("\n done!")

    def calc_flll(self, ells, r=None, verbose=False):

        '''
            calculate the f-3l tensor
        '''

        ells = numpy.array(ells)
        r = numpy.array(r)
        if not self.do_rsd:
            # case r0 -> 0, r0' -> s
            if (r[0]==0) & (r[1]==0):
                ans = (self.Pk *
                      (self.jn_s[ells[2]]) *
                       self.kk**2)

            # case r0 -> 0, r_i
            elif (r[0]==0) & (r[1]!=0):
                i2 = numpy.where(self.rr == r[1])[0]
                ans = (self.Pk *
                      (self.jn_bar_r[ells[1]][int(i2),:]) *
                      (self.jn_s[ells[2]]) *
                       self.kk**2)

            # case r_i, r_0' -> 0
            elif (r[0]!=0) & (r[1]==0):
                i1 = numpy.where(self.rr == r[0])[0]
                ans = (self.Pk *
                      (self.jn_bar_r[ells[0]][int(i1),:]) *
                      (self.jn_s[ells[2]]) *
                       self.kk**2)

            # case r_i, r_j
            elif (r[0]!=0) & (r[1]!=0):
                i1 = numpy.where(self.rr == r[0])[0]
                i2 = numpy.where(self.rr == r[1])[0]
                ans = (self.Pk *
                      (self.jn_bar_r[ells[0]][int(i1),:]) *
                      (self.jn_bar_r[ells[1]][int(i2),:]) *
                      (self.jn_s[ells[2]]) *
                       self.kk**2)

        elif self.do_rsd:
            ans = (self.Pk_ell[ells[0]] *
                  (self.jn_bar[ells[1]][i1,:]) *
                  (self.jn_bar[ells[2]][i2,:]) *
                   self.kk**2)

        if verbose:
            print("ells_unit", ells_unit)

        return ans

    def io_flll(self, action='load', fname=None):

        '''
            calculate and save/load the f-3l tensor

            default loading/saving file:
                fdir = "/Users/mianoctobers/Projects/NPCF/results/kernels_4pcf_cov_model/"
                fname = fdir + "flll_iso_4pcf.pkl"
        '''

        import pickle, itertools

        if action == 'save':

            rs = []
            r_all = numpy.insert(self.rr, 0, 0)
            for ir1 in r_all:
                for ir1_p in r_all:
                    rs.append([ir1, ir1_p])
            print(">> loading all possible ells ...")
            self.get_all_possible_lll()
            print("ells loaded!")
            ells123 = self.ell123.copy()
            flll_dict = {}
            for il, ell in enumerate(ells123):
                for ir, rr in enumerate(rs):
                    key_ell = "".join(map(str, ell))
                    key_r   = ",".join(["{:.1f}".format(x) for x in rr])
                    key = ",".join([key_ell, key_r])
                    flll_dict[key] = numpy.zeros(self.num_s)

            ic = 0
            for il, ell in enumerate(ells123):
                for ii, rr in enumerate(rs):
                    yint = numpy.sum(self.calc_flll(ell, rr), axis=1)
                    key_ell = "".join(map(str, ell))
                    key_r   = ",".join(["{:.1f}".format(x) for x in rr])
                    key = ",".join([key_ell, key_r])
                    flll_dict[key] = yint * self.dkk/(2*numpy.pi**2)
                    if ic%400==0: print(ic)
                    ic += 1

            f = open(fname, "wb")
            pickle.dump(flll_dict, f)
            f.close()

        elif action == 'load':

            if fname is None:
                raise Exception("Please provide dir to load GLs coefficient table")
            else:
                self.flll = pickle.load(open(fname, "rb"))


    def wrap_ells_fc(self, ells, which_group):

        '''
            wrap angular momenta orders for fully coupled pieces
            the output is used to call f_lll tensor
            ells (l1, l2, l3) as first element permute like:
            123
            213
            132
            321
            312
            231

            ells' is kept fixed 0, l1', l2' l3'
            ells'' according to ell and ell', especially depends
            on where zero sits.
        '''

        l1, l2, l3, l1_p, l2_p, l3_p, l1_pp, l2_pp, l3_pp = ells

        if which_group == 'block1_case1':
            ell0 = [0, 0, 0]
            ell1 = [l1, l1_p, l1_pp]
            ell2 = [l2, l2_p, l2_pp]
            ell3 = [l3, l3_p, l3_pp]

        else:
            ell0 = [l1, 0, l1]
            ell1 = [0,  l1_p, l1_p]
            ell2 = [l2, l2_p, l2_pp]
            ell3 = [l3, l3_p, l3_pp]

        self.ells_wrapped_fc = [ell0, ell1, ell2, ell3]


    def wrap_rs_fc(self, r_in, which_group, idx_six):

        '''
            wrap angular momenta orders for fully coupled pieces
            rs follow the following permutation:

                r1 r2 r3
                r2 r1 r3
                r1 r3 r2
                r3 r2 r1
                r3 r1 r2
                r2 r3 r1
        '''

        r1, r1_p, r2, r2_p, r3, r3_p = r_in
        r1, r2, r3 = self.permute_like_bob([r1, r2, r3], idx_six)
        r1_p, r2_p, r3_p = self.permute_even([r1_p, r2_p, r3_p], which_group)

        if which_group == 'block1_case1':
            r0s = [0, 0]
            r1s = [r1, r1_p]
            r2s = [r2, r2_p]
            r3s = [r3, r3_p]

        else:
            r0s = [r1, 0]
            r1s = [0,  r1_p]
            r2s = [r2, r2_p]
            r3s = [r3, r3_p]

        self.rs_wrapped_fc = [r0s, r1s, r2s, r3s]

    def wrap_flll_keys_fc(self):

        '''
            wrap f_lll tensor's keys for fully coupled pieces
            keys are stored for 4 cases and for each permutation:

                f_l1l1'l1'' fl2l2'l2'' fl3l3'l3'' fl4l4'l4''

            1st argument is wrapped angular momentum ells (from ells_wrapped_fc)
            2nd argument is wrapped directional vector r_i (from rs_wrapped_fc)
        '''

        self.flll_keys = []
        for ii in range(0,4):
            key_ell = "".join(map(str, self.ells_wrapped_fc[ii]))
            key_r = ",".join(["{:.1f}".format(x) for x in self.rs_wrapped_fc[ii]])
            self.flll_keys.append(",".join([key_ell, key_r]))


    def check_ells_is_nonzero(self, ells, which_group, idx_six, verbose=False):
        '''
            check if the permuted ells satisfies
            triangular inequality
        '''
        l1, l1_p, l1_pp, l2, l2_p, l2_pp, l3, l3_p, l3_pp = ells

        if numpy.mod(l1+l1_p+l1_pp,2) == 1:
            if verbose:
                print("violate numpy.mod(l1+l1_p+l1_pp,2)")
            return False
        elif numpy.mod(l2+l2_p+l2_pp,2) == 1:
            if verbose:
                print("violate numpy.mod(l2+l2_p+l2_pp,2)")
            return False
        elif numpy.mod(l3+l3_p+l3_pp,2) == 1:
            if verbose:
                print("violate numpy.mod(l3+l3_p+l3_pp,2)")
            return False
        elif numpy.mod(l1_pp+l2_pp+l3_pp,2) == 1:
            if verbose:
                print("violate numpy.mod(l1_pp+l2_pp+l3_pp,2)")
            return False
        elif not (l3_pp >= abs(l1_pp-l2_pp)) & ( l3_pp <= abs(l1_pp+l2_pp)):
            return False
        elif not (l1_pp >= abs(l1-l1_p)) & ( l1_pp <= abs(l1+l1_p)):
            if verbose:
                print("violate (l1_pp >= abs(l1-l1_p)) & ( l1_pp <= abs(l1+l1_p))")
            return False
        elif not (l2_pp >= abs(l2-l2_p)) & ( l2_pp <= abs(l2+l2_p)):
            if verbose:
                print("violate (l2_pp >= abs(l2-l2_p)) & ( l2_pp <= abs(l2+l2_p))")
            return False
        elif not (l3_pp >= abs(l3-l3_p)) & ( l3_pp <= abs(l3+l3_p)):
            if verbose:
                print("violate (l3_pp >= abs(l3-l3_p)) & ( l3_pp <= abs(l3+l3_p))")
            return False
        else:
            return True


    def get_yint_fc(self, ells, ig, idx_six, ic=999):

        '''
            calculate integral for fully coupled terms
        '''

        yints = 0.
        key_ell = "".join(map(str,ells))
        key_GLs = ",".join([key_ell, ig])
        try:
            GLs = self.GLs_dict[key_GLs]
            yints = ((self.flll[self.flll_keys[0]]) *
                      (self.flll[self.flll_keys[1]]) *
                      (self.flll[self.flll_keys[2]]) *
                      (self.flll[self.flll_keys[3]])) * GLs
            if self.verbose:
                if ic == 10:
                    print("ic=10, key_GLs", key_GLs, "GLs", self.GLs_dict[key_GLs])
                    print("self.flll_keys[0]", self.flll_keys[0])
                    print("self.flll_keys[1]", self.flll_keys[1])
                    print("self.flll_keys[2]", self.flll_keys[2])
                    print("self.flll_keys[3]", self.flll_keys[3])
        except:
            raise ValueError("coefficient is not calculated!")

        self.yint = (numpy.sum(yints * (self.ss)**2, axis=0) * self.dss / self.vol)

        return self.yint

    def calc_GLs_bob(self, ells, which_group='block1_case1', idx_six=0):

        '''
            calculate GLs using Bob's notation

            ells:
                permuted l1, l1', l1'', l2, l2', l2'', l3, l3, l3''

        '''

        l1, l1_p, l1_pp, l2, l2_p, l2_pp, l3, l3_p, l3_pp = ells

        ells_6perm = self.permute_ells_fc(ells)
        l_G1, l_H1, l1_pp, l_G2, l_H2, l2_pp, l_G3, l_H3, l3_pp = ells_6perm[idx_six]

        def calc_coeff(l1, l1_p, l1_pp, l2, l2_p, l2_pp, l3, l3_p, l3_pp,
                       l_G1, l_H1, l_G2, l_H2, l_G3, l_H3,
                       which_group=None):

            phi = l1 + l1_p + l1_pp + l2 + l2_p + l2_pp + l3 + l3_p + l3_pp
            if which_group == 'block1_case1':
                phase = (-1)**(phi/2)
            else:
                phase = (-1)**(phi/2 + l_G1 + l_H1)

            G_Ls = (numpy.sqrt((2*l_G1+1)*(2*l_H1+1))*
                    numpy.sqrt((2*l_G2+1)*(2*l_H2+1))*
                    numpy.sqrt((2*l_G3+1)*(2*l_H3+1))*
                    (2*l1_pp+1)*(2*l2_pp+1)*(2*l3_pp+1)*
                    numpy.float64(wigner_3j(l_G1,l_H1,l1_pp,0,0,0))*
                    numpy.float64(wigner_3j(l_G2,l_H2,l2_pp,0,0,0))*
                    numpy.float64(wigner_3j(l_G3,l_H3,l3_pp,0,0,0))*
                    numpy.float64(wigner_3j(l1_pp,l2_pp,l3_pp,0,0,0))*
                    numpy.float64(wigner_9j(l_G1,l_G2,l_G3,
                                            l_H1,l_H2,l_H3,
                                            l1_pp,l2_pp,l3_pp)))

            return phase, G_Ls


        phase, G_Ls = calc_coeff(l1, l1_p, l1_pp, l2, l2_p, l2_pp, l3, l3_p, l3_pp,
                                 l_G1, l_H1, l_G2, l_H2, l_G3, l_H3, which_group)

        ans = ((4*numpy.pi)**4 * phase * G_Ls ).real

        return ans

    def calc_GLs_oliver(self, ells, which_group):

        '''
            calculate GLs using Oliver's notation

            ells:
                permuted l1, l1', l1'', l2, l2', l2'', l3, l3, l3''

        '''

        l_G1, l_G2, l_G3, l_H1, l_H2, l_H3, l1_pp, l2_pp,l3_pp = ells

        def calc_coeff(l_G1, l_H1, l1_pp, l_G2, l_H2, l2_pp, l_G3, l_H3, l3_pp,
                       which_group):

            if which_group == 'block1_case1':
                phi = -l_G1 - l_H1 + l1_pp - l_G2 - l_H2 + l2_pp - l_G3 - l_H3 + l3_pp
            else:
                phi = l2_pp + l3_pp - l_H2 - l_H3 - l_G2 - l_G3

            phase = (-1)**(phi/2)

            G_Ls = (numpy.sqrt((2*l_G1+1)*(2*l_H1+1))*
                    numpy.sqrt((2*l_G2+1)*(2*l_H2+1))*
                    numpy.sqrt((2*l_G3+1)*(2*l_H3+1))*
                    (2*l1_pp+1)*(2*l2_pp+1)*(2*l3_pp+1)*
                    numpy.float64(wigner_3j(l_G1,l_H1,l1_pp,0,0,0))*
                    numpy.float64(wigner_3j(l_G2,l_H2,l2_pp,0,0,0))*
                    numpy.float64(wigner_3j(l_G3,l_H3,l3_pp,0,0,0))*
                    numpy.float64(wigner_3j(l1_pp,l2_pp,l3_pp,0,0,0))*
                    numpy.float64(wigner_9j(l_G1,l_G2,l_G3,
                                            l_H1,l_H2,l_H3,
                                            l1_pp,l2_pp,l3_pp)))

            return phase, G_Ls

        phase, G_Ls = calc_coeff(l_G1, l_H1, l1_pp, l_G2, l_H2, l2_pp, l_G3, l_H3, l3_pp,
                               which_group)

        ans = (4*numpy.pi)**4 * phase * G_Ls

        return ans


    def io_GLs(self, action='load', fname=None, verbose=False):

        '''
            default loading/saving file:
                fdir = "/Users/mianoctobers/Projects/NPCF/results/kernels_4pcf_cov_model/"
                fname = fdir + "coeffs_LG_4pcf_block1.pkl"
        '''

        import pickle

        if action == 'save':

            groups = ['block1_case1','block1_case2','block1_case3','block1_case4']
            ic = 0
            keys = []
            GLs_dict = {}
            for ii, iell_g in enumerate(self.ells_g_comb2):
                l1, l2, l3 = [int(i) for i in iell_g[0]]
                l1_p, l2_p, l3_p = [int(i) for i in iell_g[1]]
                lls = [l1, l2, l3]
                lls_p = [l1_p, l2_p, l3_p]
                for gg, ig in enumerate(groups):
                    for idx_six in range(0,6):
                        l_Gs, l_Hs = self.permute_ell_GH(lls, lls_p, ig, idx_six)
                        ll_pps = self.get_lambda_pp_noperm(l_Gs, l_Hs)
                        for jj, iell in enumerate(ll_pps):
                            ell_perm = l_Gs + l_Hs + iell
                            key_ell = "".join(map(str, ell_perm))
                            if key_ell not in keys:
                                G_Ls = self.calc_GLs_oliver(ells=ell_perm, which_group=ig)
                                GLs_dict[key_ell] = G_Ls
                                keys.append(key_ell)

                    if verbose:
                        if ic%200 == 0: print(ic)
                        ic+=1

            f = open(fname, "wb")
            pickle.dump(GLs_dict, f)
            f.close()

        elif action == 'load':
            if fname is None:
                raise Exception("Please provide dir to load GLs coefficient table")
            else:
                self.GLs_dict = pickle.load(open(fname, "rb"))


    def get_4pcf_cov_model(self, idx_min=0, idx_max=1, groups=None, do_save=False):

        '''
            Main code that does the 4PCF covariance calculation
                idx_min: lower index for ell order
                idx_max: upper index for ell order
        '''

        print(">> calculate 4PCF ...")
        self.cov_dict = {}
        if groups == None:
            groups = ['block1_case1','block1_case2','block1_case3','block1_case4']

        for ii, iell_g in enumerate(self.ells_g_comb2[idx_min:idx_max]):
            l1, l2, l3 = [int(i) for i in iell_g[0]]
            l1_p, l2_p, l3_p = [int(i) for i in iell_g[1]]
            lls = [l1, l2, l3]
            lls_p = [l1_p, l2_p, l3_p]
            print("l1, l2, l3, l1_p, l2_p, l3_p = ", lls, lls_p)

            ic = 0
            self.rbins = []
            ndim = len(self.meta_data.bins)
            self.y_arr_24 = numpy.zeros([ndim,ndim,24])
            self.yint_arr = numpy.zeros([ndim, ndim])

            for ir1 in self.rr[:-2]:
                for ir2 in self.rr[self.rr>ir1][:-1]:
                    for ir3 in self.rr[self.rr>ir2]:
                        for ir1_p in self.rr[:-2]:
                            for ir2_p in self.rr[self.rr>ir1_p][:-1]:
                                for ir3_p in self.rr[self.rr>ir2_p]:
                                    rbin_i, rbin_j = ic//ndim, ic%ndim
                                    if ic%2000==0: print(int(ic//2000)*"*"+"%.f"%(ic/ndim**2*100)+"%","ic=",ic, end="\r")
                                    ic += 1
                                    r_in = [ir1, ir1_p, ir2, ir2_p, ir3, ir3_p]
                                    for gg, ig in enumerate(groups):
                                        for idx_six in range(0,6):
                                            self.wrap_rs_fc(r_in, which_group=ig, idx_six=idx_six)
                                            l_Gs, l_Hs = self.permute_ell_GH(lls, lls_p, ig, idx_six)
                                            ll_pps = self.get_lambda_pp_noperm(l_Gs, l_Hs)
                                            for jj, iell in enumerate(ll_pps):
                                                l1_pp, l2_pp, l3_pp = iell
                                                ell_perm = l_Gs + l_Hs + iell
                                                self.wrap_ells_fc(ell_perm, which_group=ig)
                                                self.wrap_flll_keys_fc()
                                                yint = self.get_yint_fc(ell_perm, ig, idx_six, ic)
                                                self.yint_arr[rbin_i, rbin_j] += yint
                                                self.y_arr_24[rbin_i, rbin_j, gg*6+idx_six] += yint


            key = (str(l1)+str(l2)+str(l3), str(l1_p)+str(l2_p)+str(l3_p))
            self.cov_dict[key] = self.yint_arr.copy()

            if do_save:
                fname = "/Users/mianoctobers/Projects/NPCF/cov/4pcf_real_z2_full_couple_%sx%s.cov"%(key[0],key[1])
                f = open(fname, "wb")
                pickle.dump(self.cov_dict[key], f)
                f.close()

#            cov_T = self.cov_dict[key].T.copy()
#            numpy.fill_diagonal(cov_T, 0)
#            self.cov_dict[key] += cov_T

def get_cnorm(cov, diag0=None, diag1=None):

    '''
        Calculate correlation matrix for input
            "cov"
        If diag0, diag1 are given, then normalise
        the covariance according to those diagonals
    '''


    ndim = len(cov)
    cnorm = numpy.zeros([ndim, ndim])
    for ii in range(ndim):
        for jj in range(ndim):
            if diag0 is None:
                cnorm[ii,jj] = cov[ii,jj]/numpy.sqrt(cov[ii,ii]*cov[jj,jj])
            else:
                cnorm[ii,jj] = cov[ii,jj]/(diag0[ii]*diag1[jj])
    return cnorm

def get_memory_size(array):
    print("memory of array = ", array.size * array.itemsize / 1e6, "Mb")


def estimate_time(cov_model_4pcf_zs):
    ll_pps_list = []
    ic = 0
    for ii, iell_g in enumerate(cov_model_4pcf_zs.ells_g_comb2):
        l1, l2, l3 = [int(i) for i in iell_g[0]]
        l1_p, l2_p, l3_p = [int(i) for i in iell_g[1]]
        lls = [l1, l2, l3]
        lls_p = [l1_p, l2_p, l3_p]
        ## calculate all \Lambda'' allowed by
        ## triangulation rules
        if '5' in ('').join(iell_g):
            continue

        ll_pps_list.append(cov_model_4pcf_zs.get_lambda_pp(lls, lls_p))
        ic+=len(cov_model_4pcf_zs.get_lambda_pp(lls, lls_p))
    print(ic)
    #     print(self.ells_g_comb2[ii], len(ll))
    print("Iterations for all orders =", ic, ", Number of days =", ic*15/60)
