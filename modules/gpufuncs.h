// NBIN is the number of bins we'll sort the radii into. Must be at least N-1 for the N-point function
#define NBIN 20
// THREADS_PER_BLOCK is the number of threads in a block - default is 512 
#define THREADS_PER_BLOCK 512
// DISCON2_PARTICLES_PER_THREAD defines the number of particles accumulated per thread
#define DISCON2_PARTICLES_PER_THREAD 1000

// alm accumulation code
void accumulate_multipoles(double *d_mult, int *d_mult_ct, double *x_array, double *y_array,
                           double *z_array, double *w_array, int *bin_array,
                          int length, int max_length, int nmult, int order);

// alm accumulation code
void accumulate_multipoles2(double *d_mult, int *d_mult_ct, double *x_array, double *y_array,
                           double *z_array, double *w_array, int *bin_array,
                          int length, int max_length, int nmult, int order);

void copy_mult(double **p_mult, double *mult, int **p_mult_ct, int *mult_ct, int size, int size_ct);

void gpu_free_mult(double *mult, int *mult_ct);

void gpu_allocate_mult(double **p_mult, double *mult, int **p_mult_ct, int *mult_ct, int size, int size_ct);


// ======================================================= /
//  ALL ADD_TO_POWER FUNCTIONS ARE HERE                    /
// ======================================================= /


//3PCF kernels
//Only 1 kernel option for 3CF, 3 precision modes
void gpu_add_to_power3_orig(double *d_threepcf, double *d_weight3pcf,
        double *weights, int *lut3_i, int *lut3_j, int *lut3_ct, 
        int nb, int nlm, int nouter, int order, int np);

void gpu_add_to_power3_orig_float(float *d_threepcf, float *d_weight3pcf,
        double *weights, int *lut3_i, int *lut3_j, int *lut3_ct,
        int nb, int nlm, int nouter, int order, int np);

void gpu_add_to_power3_orig_mixed(double *d_threepcf, double *d_weight3pcf,
        double *weights, int *lut3_i, int *lut3_j, int *lut3_ct,
        int nb, int nlm, int nouter, int order, int np);

// ======================================================= /

//DISCONNECTED 4PCF kernels
//Only 1 kernel option for DISCONNECTED 4PCF, 3 precision modes
void gpu_add_to_power_discon1_orig(double *d_discon1_r, double *d_discon1_i,
	double *d_weightdiscon, double *weights,
	int *lut_discon_ell, int *lut_discon_mm, 
        int nb, int nlm, int ndiscon1, int order, int np);

void gpu_add_to_power_discon1_orig_float(float *f_discon1_r,
	float *f_discon1_i, float *f_weightdiscon,
	double *weights, int *lut_discon_ell,
	int *lut_discon_mm, int nb, int nlm, int ndiscon1, int order, int np);

void gpu_add_to_power_discon1_orig_mixed(double *d_discon1_r,
	double *d_discon1_i, double *d_weightdiscon,
	double *weights, int *lut_discon_ell,
	int *lut_discon_mm, int nb, int nlm, int ndiscon1, int order, int np);

void gpu_add_to_power_discon2_orig(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double *weights, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int nb, int nlm, int nouter, int order, int ninner, int np);

void gpu_add_to_power_discon2_b(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double wp, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner);

void gpu_add_to_power_discon2_final(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double *weights, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner, int np,
	int qbalance, int qinvert);

void gpu_add_to_power_discon2_final_float(float *f_discon2_r, float *f_discon2_i,
        float *f_weightdiscon, double *weights, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner, int np,
	int qbalance, int qinvert);

void gpu_add_to_power_discon2_final_mixed(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double *weights, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner, int np,
	int qbalance, int qinvert);

// ======================================================= /

//4PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes
//run main kernel gpu == 1
void gpu_add_to_power4(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nb, int nlm, int nouter, int ninner, int nell4);

//float version of main kernel
void gpu_add_to_power4_float(float *d_fourpcf, float *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4);

//mixed precision
void gpu_add_to_power4_mixed(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4);

//alternate (original) kernel
void gpu_add_to_power4_orig(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
	int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nb, int nlm, int nouter, int ninner, int nell4);

//float version
void gpu_add_to_power4_orig_float(float *d_fourpcf, float *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4);

//mixed precision
void gpu_add_to_power4_orig_mixed(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4);

// ======================================================= /

//5PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes
//run main kernel gpu == 1
void gpu_add_to_power5(double *d_fivepcf, double *d_weight5pcf, 
	int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
	int *lut5_l4, bool *lut5_odd, int *lut5_n,
	int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int nlm, int nouter, int ninner, int nell5);

//float version of main kernel
void gpu_add_to_power5_float(float *d_fivepcf, float *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5);

//mixed precision
void gpu_add_to_power5_mixed(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5);

//alternate (original) kernel
void gpu_add_to_power5_orig(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int nlm, int nouter, int ninner, int nell5);

//float version
void gpu_add_to_power5_orig_float(float *d_fivepcf, float *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5);

//mixed precision
void gpu_add_to_power5_orig_mixed(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5);

// ======================================================= /

//6PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes
//run main kernel gpu == 1
void gpu_add_to_power6(double *d_sixpcf, double *d_weight6pcf,
        int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
	int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
	int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
	int *lut6_k, int *lut6_l, int *lut6_m,
        double wp, int nb, int nlm, int nouter, int ninner, int nell6);

//float version of main kernel
void gpu_add_to_power6_float(float *d_sixpcf, float *d_weight6pcf,
        int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
        int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
	int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
	int *lut6_k, int *lut6_l, int *lut6_m,
        float wp, int nb, int nlm, int nouter, int ninner, int nell6);

//mixed precision
void gpu_add_to_power6_mixed(double *d_sixpcf, double *d_weight6pcf,
        int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
        int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
        int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
        int *lut6_k, int *lut6_l, int *lut6_m,
        float wp, int nb, int nlm, int nouter, int ninner, int nell6);


// ======================================================= /
//  ALL MEMORY FUNCTIONS ARE HERE                          /
// ======================================================= /
//allocate LUTs used in all kernels
//3PCF LUTs
void gpu_allocate_luts3(int **p_lut3_i, int **p_lut3_j, int **p_lut3_ct, int nouter);

void gpu_allocate_threepcf(double **p_threepcf, double *threepcf, int size);

void gpu_allocate_weight3pcf(double **p_weight3pcf, double *weight3pcf, int size);

void copy_threepcf(double **p_threepcf, double *threepcf, int size); 

void gpu_allocate_threepcf(float **p_threepcf, double *threepcf, int size);

void gpu_allocate_weight3pcf(float **p_weight3pcf, double *weight3pcf, int size);

void copy_threepcf(float **p_threepcf, double *threepcf, int size);

// ======================================================= /

//free memory
void gpu_free_luts3(int *lut3_i, int *lut3_j, int *lut3_ct);

void gpu_free_memory3(double *threepcf, double *weight3pcf);

void gpu_free_memory3(float *threepcf, float *weight3pcf);

// ======================================================= /

//DISCONNECTED 4PCF LUTs
void gpu_allocate_luts_discon1(int **p_lut_discon_ell, int **p_lut_discon_mm,
	int size);

void gpu_allocate_discon1(double **p_discon1_r, double **p_discon1_i,
	std::complex<double> *discon1, int size);

void gpu_allocate_weightdiscon(double **p_weightdiscon, double *weightdiscon,
	int size);

void copy_discon1(double **p_discon1_r, double **p_discon1_i,
	std::complex<double> *discon1, int size);

void gpu_allocate_discon1(float **p_discon1_r, float **p_discon1_i,
        std::complex<double> *discon1, int size);

void gpu_allocate_weightdiscon(float **p_weightdiscon, double *weightdiscon,
	int size);

void copy_discon1(float **p_discon1_r, float **p_discon1_i,
	std::complex<double> *discon1, int size);

//DISCON2 term
void gpu_allocate_luts_discon2(int **p_lut_discon_ell1,
	int **p_lut_discon_ell2, int **p_lut_discon_mm1,
	int **p_lut_discon_mm2, int size);

void gpu_allocate_luts_discon2_inner(int **p_lut_discon_i,
	int **p_lut_discon_j, int size);

void gpu_allocate_discon2(double **p_discon2_r, double **p_discon2_i,
        std::complex<double> *discon2, int size);

void copy_discon2(double **p_discon2_r, double **p_discon2_i,
        std::complex<double> *discon2, int size);

void gpu_allocate_discon2(float **p_discon2_r, float **p_discon2_i,
        std::complex<double> *discon2, int size);

void copy_discon2(float **p_discon2_r, float **p_discon2_i,
        std::complex<double> *discon2, int size);

// ======================================================= /

//free memory
void gpu_free_luts_discon1(int *lut_discon_ell, int *lut_discon_mm);

void gpu_free_memory_discon1(double *d_discon1_r, double *d_discon1_i,
	double *weightdiscon);

void gpu_free_memory_discon1(float *f_discon1_r, float *f_discon1_i,
	float *weightdiscon);

//DISCON2 term
void gpu_free_luts_discon2(int *lut_discon_ell1, int *lut_discon_ell2,
	int *lut_discon_mm1, int *lut_discon_mm2, int *lut_discon_i,
	int *lut_discon_j);

// ======================================================= /

//4PCF LUTs
void gpu_allocate_luts4(int **p_lut4_l1, int **p_lut4_l2, int **p_lut4_l3,
	bool **p_lut4_odd, int **p_lut4_n, int **p_lut4_zeta, int **p_lut4_i,
	int **p_lut4_j, int **p_lut4_k, int nouter, int ninner);

//allocate m LUTs for alternate kernels
void gpu_allocate_m_luts4(int **p_lut4_m1, int **p_lut4_m2, int nouter);

//allocate and copy fourpcf
void gpu_allocate_fourpcf(double **p_fourpcf, double *fourpcf, int size);

void gpu_allocate_fourpcf(float **p_fourpcf, double *fourpcf, int size);

//allocate and copy weight4pcf
void gpu_allocate_weight4pcf(double **p_weight4pcf, double *weight4pcf, int size);

void gpu_allocate_weight4pcf(float **p_weight4pcf, double *weight4pcf, int size);

//copy device array back to host for no memcpy
void copy_fourpcf(double **p_fourpcf, double *fourpcf, int size);

void copy_fourpcf(float **p_fourpcf, double *fourpcf, int size);

// ======================================================= /

//free memory 4

void gpu_free_luts4(int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k);

void gpu_free_memory4(double *fourpcf, double *weight4pcf);

void gpu_free_memory4(float *fourpcf, float *weight4pcf);

void gpu_free_memory_m4(int *lut4_m1, int *lut4_m2);

// ======================================================= /
//5PCF LUTs

//allocate LUTs used in all kernels
void gpu_allocate_luts(int **p_lut5_l1, int **p_lut5_l2, int **p_lut5_l12,
	int **p_lut5_l3, int **p_lut5_l4, bool **p_lut5_odd, int **p_lut5_n,
        int **p_lut5_zeta, int **p_lut5_i, int **p_lut5_j, int **p_lut5_k,
	int **p_lut5_l, int nouter, int ninner);

//allocate m LUTs for alternate kernels
void gpu_allocate_m_luts(int **p_lut5_m1, int **p_lut5_m2, int **p_lut5_m3, int nouter);

//allocate and copy fivepcf
void gpu_allocate_fivepcf(double **p_fivepcf, double *fivepcf, int size);

void gpu_allocate_fivepcf(float **p_fivepcf, double *fivepcf, int size);

//allocate and copy weight5pcf
void gpu_allocate_weight5pcf(double **p_weight5pcf, double *weight5pcf, int size);

void gpu_allocate_weight5pcf(float **p_weight5pcf, double *weight5pcf, int size);

//copy device array back to host for no memcpy
void copy_fivepcf(double **p_fivepcf, double *fivepcf, int size);

void copy_fivepcf(float **p_fivepcf, double *fivepcf, int size);

// ======================================================= /

//free memory
void gpu_free_luts(int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n, int *lut5_zeta, int *lut5_i,
        int *lut5_j, int *lut5_k, int *lut5_l);

void gpu_free_memory(double *fivepcf, double *weight5pcf);

void gpu_free_memory(float *fivepcf, float *weight5pcf);

void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3);

// ======================================================= /

//6PCF LUTs

//allocate LUTs used in all kernels
void gpu_allocate_luts6(int **p_lut6_l1, int **p_lut6_l2, int **p_lut6_l12,
        int **p_lut6_l3, int **p_lut6_l123, int **p_lut6_l4,
	int **p_lut6_l5, bool **p_lut6_odd, int **p_lut6_n,
        int **p_lut6_zeta, int **p_lut6_i, int **p_lut6_j, int **p_lut6_k,
        int **p_lut6_l, int **p_lut6_m, int nouter, int ninner);

//allocate and copy sixpcf
void gpu_allocate_sixpcf(double **p_sixpcf, double *sixpcf, int size);

void gpu_allocate_sixpcf(float **p_sixpcf, double *sixpcf, int size);

//allocate and copy weight6pcf
void gpu_allocate_weight6pcf(double **p_weight6pcf, double *weight6pcf, int size);

void gpu_allocate_weight6pcf(float **p_weight6pcf, double *weight6pcf, int size);

//copy device array back to host for no memcpy
void copy_sixpcf(double **p_sixpcf, double *sixpcf, int size);

void copy_sixpcf(float **p_sixpcf, double *sixpcf, int size);

// ======================================================= /

//free memory
void gpu_free_luts6(int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
	int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
	int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
	int *lut6_k, int *lut6_l, int *lut6_m);

void gpu_free_memory6(double *sixpcf, double *weight6pcf);

void gpu_free_memory6(float *sixpcf, float *weight6pcf);

// ======================================================= /
//  ALL ALM FUNCTIONS ARE HERE                             /
// ======================================================= /

//allocate alms
void gpu_allocate_alms(int np, int nb, int nlm, bool isDouble);

void gpu_compute_alms(int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult);

void gpu_compute_alms_float(int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult);

void gpu_free_memory_alms(bool isDouble);

// ======================================================= /
//  PAIRS AND MULTIPOLES ACCUMULATION                      /
// ======================================================= /

//memory operation

void gpu_allocate_multipoles(double **p_msave, int **p_csave,
        int **p_pnum, int **p_spnum, int **p_snp, int **p_sc,
        int nmult, int nbin, int np, int nmax);

void gpu_allocate_multipoles_fast(double **p_msave, int **p_csave,
        int **p_start_list, int **p_np_list, int **p_cellnums,
        int nmult, int nbin, int np, int maxp, int nmax, int nc);

void gpu_allocate_particle_arrays(double **p_posx, double **p_posy, double **p_posz, double **p_weights, int np);

void gpu_allocate_pair_arrays(double **p_x0i, double **p_x2i, int nbin);

void gpu_allocate_periodic(int **p_delta_x, int **p_delta_y, int ** p_delta_z, int nmax);

void free_gpu_multipole_arrays(double *msave, int *csave,
        int *pnum, int *spnum, int *snp, int *sc,
        double *posx, double *posy, double *posz,
        double *weights, double *x0i, double *x2i);

void free_gpu_periodic_arrays(int *delta_x, int *delta_y, int *delta_z);

// ======================================================= /

// We have 4 methods for each of pairs_only and pairs_and_multipoles
// DEFAULTS in CAPS
// NORMAL mode or periodic
// original kernel or FAST

// Additionally each method takes 2 bools which cascade to 4x kernels
// SHARED (default TRUE) = SHARED vs global memory
// gpufloat (default FALSE) = DOUBLE vs float

// Thus we have 16 kernels for each bound up in 4 methods.

//PAIRS ONLY

void gpu_add_pairs_only(double *posx, double *posy, double *posz, double *w,
        int *pnum, int *spnum, int *snp, int *sc, double *x0i, double *x2i,
        int n, int nbin, float rmin, float rmax, bool shared, bool gpufloat);

void gpu_add_pairs_only_periodic(double *posx, double *posy, double *posz,
        double *w, int *pnum, int *spnum, int *snp, int *sc, double *x0i,
        double *x2i, int *delta_x, int *delta_y, int *delta_z, int n,
        int nbin, float rmin, float rmax, double cellsize, bool shared,
        bool gpufloat);
        
void gpu_add_pairs_only_fast(double *posx, double *posy, double *posz,
	double *w, int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order,
        int nmult, float rmin, float rmax, int nside_cuboid_x,
        int nside_cubiod_y, int nside_cuboid_z, int ncells, int maxsep,
        int pstart, bool shared, bool gpufloat);

void gpu_add_pairs_only_periodic_fast(double *posx, double *posy, double *posz,
        double *w, int *start_list, int *np_list, int *cellnums, double *x0i,
        double *x2i, int n, int nbin, int order, int nmult, float rmin,
        float rmax, int nside_cuboid_x, int nside_cubiod_y,
        int nside_cuboid_z, int ncells, int maxsep, int pstart,
	double cellsize, bool shared, bool gpufloat);
        
//PAIRS AND MULTIPOLES

void gpu_add_pairs_and_multipoles(double *m, double *posx, double *posy,
        double *posz, double *w, int *ct, int *pnum, int *spnum,
        int *snp, int *sc, double *x0i, double *x2i, int n, int nbin,
	int order, int nmult, float rmin, float rmax, int pstart,
	bool shared, bool gpufloat);

void gpu_add_pairs_and_multipoles_periodic(double *m, double *posx,
	double *posy, double *posz, double *w, int *ct, int *pnum, int *spnum,
        int *snp, int *sc, double *x0i, double *x2i, int *delta_x,
	int *delta_y, int *delta_z, int n, int nbin, int order, int nmult,
	float rmin, float rmax, int pstart, double cellsize, bool shared,
	bool gpufloat);

void gpu_add_pairs_and_multipoles_fast(double *m, double *posx, double *posy,
        double *posz, double *w, int *ct, int *start_list,
        int *np_list, int *cellnums, double *x0i, double *x2i,
        int n, int nbin, int order, int nmult, float rmin, float rmax,
        int nside_cuboid_x, int nside_cubiod_y, int nside_cuboid_z,
        int ncells, int maxsep, int pstart, bool shared, bool gpufloat);

void gpu_add_pairs_and_multipoles_periodic_fast(double *m, double *posx,
        double *posy, double *posz, double *w, int *ct, int *start_list,
        int *np_list, int *cellnums, double *x0i, double *x2i, int n,
        int nbin, int order, int nmult, float rmin, float rmax,
        int nside_cuboid_x, int nside_cubiod_y, int nside_cuboid_z,
        int ncells, int maxsep, int pstart, double cellsize,
        bool shared, bool gpufloat);

// ======================================================= /
// call cudaDeviceSynchronize

void gpu_device_synchronize();

void gpu_print_cuda_error();
