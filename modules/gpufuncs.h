
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

//allocate LUTs used in all kernels
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

//allocate alms
void gpu_allocate_alms(int np, int nb, int nlm, bool isDouble);

//copy device array back to host for no memcpy
void copy_fourpcf(double **p_fourpcf, double *fourpcf, int size);

void copy_fourpcf(float **p_fourpcf, double *fourpcf, int size);

//free memory
void gpu_free_luts4(int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k);

void gpu_free_memory4(double *fourpcf, double *weight4pcf);

void gpu_free_memory4(float *fourpcf, float *weight4pcf);

void gpu_free_memory_m4(int *lut4_m1, int *lut4_m2);


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

//free memory
void gpu_free_luts(int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n, int *lut5_zeta, int *lut5_i,
	int *lut5_j, int *lut5_k, int *lut5_l);

void gpu_free_memory(double *fivepcf, double *weight5pcf);

void gpu_free_memory(float *fivepcf, float *weight5pcf);

void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3);

void gpu_free_memory_alms(bool isDouble);

void gpu_allocate_multipoles(double **p_msave, int **p_csave,
        int **p_pnum, int **p_spnum, int **p_snp, int **p_sc,
        int nmult, int nbin, int np, int nmax);

void gpu_allocate_particle_arrays(double **p_posx, double **p_posy, double **p_posz, double **p_weights, int np);

void gpu_allocate_pair_arrays(double **p_x0i, double **p_x2i, int nbin);

void gpu_allocate_periodic(int **p_delta_x, int **p_delta_y, int ** p_delta_z, int nmax);

void free_gpu_multipole_arrays(double *msave, int *csave,
        int *pnum, int *spnum, int *snp, int *sc,
        double *posx, double *posy, double *posz,
        double *weights, double *x0i, double *x2i);

void free_gpu_periodic_arrays(int *delta_x, int *delta_y, int *delta_z);

void gpu_add_pairs_and_multipoles(double *m, double *posx, double *posy,
        double *posz, double *w, int *ct, int *pnum, int *spnum,
        int *snp, int *sc, double *x0i, double *x2i, int n, int nbin,
	int order, int nmult, float rmin, float rmax, int pstart);

void gpu_add_pairs_and_multipoles_periodic(double *m, double *posx,
	double *posy, double *posz, double *w, int *ct, int *pnum, int *spnum,
        int *snp, int *sc, double *x0i, double *x2i, int *delta_x,
	int *delta_y, int *delta_z, int n, int nbin, int order, int nmult,
	float rmin, float rmax, int pstart, double cellsize);

void gpu_compute_alms(int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult);

void gpu_compute_alms_float(int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult);

void gpu_device_synchronize();
