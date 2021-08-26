import time, importlib, sys
import model_4PCF_Gaussian_covariance_isotropic_basis
importlib.reload(model_4PCF_Gaussian_covariance_isotropic_basis)
#from model_4PCF_Gaussian_covariance_isotropic_basis import model_cov_4PCF, get_cnorm
from model_4PCF_Gaussian_covariance_isotropic_basis_auto_perm import model_cov_4PCF, get_cnorm
import NPCF_utils
from NPCF_utils import GetCoeffCpp
​
idx_min = int(sys.argv[1])
idx_max = int(sys.argv[2])
fdir = "/global/homes/o/octobers/Projects/NPCF/results/"
npcf = 4
meta_data = GetCoeffCpp(fdir, npcf)
meta_data.load_lognormal_4pcf_1000()
​
start_time = time.time()
Cov_Model_4pcf = model_cov_4PCF(meta_data=meta_data, fname_f3l="flll_iso_4pcf_27x153.pkl", fname_GLs="coeffs_LG_4pcf_block1_27x153.pkl",
                                k_in=meta_data.k_in, Pk_in=meta_data.pk0_in, shotnoise=meta_data.shot_noise, r_in=meta_data.rbins_1d)
​
Cov_Model_4pcf.init_arrs()
Cov_Model_4pcf.init_2stat()
Cov_Model_4pcf.get_ells(verbose=False)
Cov_Model_4pcf.gen_jnbar_dict(verbose=False)
#Cov_Model_4pcf.io_flll(fname="/global/homes/o/octobers/Projects/NPCF/results/kernels_4pcf_cov_model/flll_iso_4pcf_logn_real_27x153.pkl")
#Cov_Model_4pcf.io_GLs(action='load', fname="/global/homes/o/octobers/Projects/NPCF/results/kernels_4pcf_cov_model/coeffs_LG_4pcf_block1_27x153.pkl.oliver")
Cov_Model_4pcf.io_flll(fname="/global/homes/o/octobers/Projects/NPCF/results/kernels_4pcf_cov_model/flll_iso_4pcf_logn_real_55ells_27x153.pkl")
Cov_Model_4pcf.io_GLs(action='load', fname="/global/homes/o/octobers/Projects/NPCF/results/kernels_4pcf_cov_model/coeffs_LG_4pcf_full_couple_27x153.pkl.oliver")
Cov_Model_4pcf.get_4pcf_cov_model(idx_min=idx_min, idx_max=idx_max, do_save=True)
​
print("elapsed time:", time.time()-start_time, 's')
