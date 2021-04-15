typedef std::complex<double> Complex;

//run main kernel gpu == 1
void gpu_add_to_power4(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4);

//float version of main kernel
void gpu_add_to_power4_float(float *d_fourpcf, float *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4);

//mixed precision
void gpu_add_to_power4_mixed(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4);

//alternate (original) kernel
void gpu_add_to_power4_orig(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_odd,
	int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4);

//float version
void gpu_add_to_power4_orig_float(float *d_fourpcf, float *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_odd,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4);

//mixed precision
void gpu_add_to_power4_orig_mixed(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_odd,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4);

//run main kernel gpu == 1
void gpu_add_to_power5(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
	Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
	int *lut5_l4,  int *lut5_odd, int *lut5_n,
	int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//float version of main kernel
void gpu_add_to_power5_float(float *d_fivepcf, float *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4,  int *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//mixed precision
void gpu_add_to_power5_mixed(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//with memcpy
void gpu_add_to_power5_with_memcpy(double *fivepcf, double *weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//alternate (original) kernel
void gpu_add_to_power5_orig(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//float version
void gpu_add_to_power5_orig_float(float *d_fivepcf, float *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//mixed precision
void gpu_add_to_power5_orig_mixed(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//with memcpy
void gpu_add_to_power5_orig_with_memcpy(double *fivepcf, double *weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

//allocate LUTs used in all kernels
void gpu_allocate_luts4(int **p_lut4_l1, int **p_lut4_l2, int **p_lut4_l3,
        int **p_lut4_n, int **p_lut4_odd, int **p_lut4_zeta, int **p_lut4_i, int **p_lut4_j, int **p_lut4_k,
        int nouter, int ninner);

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

//free memory
void gpu_free_luts4(int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k);

void gpu_free_memory4(double *fourpcf, double *weight4pcf);

void gpu_free_memory4(float *fourpcf, float *weight4pcf);

void gpu_free_memory_m4(int *lut4_m1, int *lut4_m2);


//allocate LUTs used in all kernels
void gpu_allocate_luts(int **p_lut5_l1, int **p_lut5_l2, int **p_lut5_l12, int **p_lut5_l3,
        int **p_lut5_l4, int **p_lut5_odd, int **p_lut5_n,
        int **p_lut5_zeta, int **p_lut5_i, int **p_lut5_j, int **p_lut5_k, int **p_lut5_l,
        int nouter, int ninner);

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
        int *lut5_l4, int *lut5_odd, int *lut5_n, int *lut5_zeta, int *lut5_i,
	int *lut5_j, int *lut5_k, int *lut5_l);

void gpu_free_memory(double *fivepcf, double *weight5pcf);

void gpu_free_memory(float *fivepcf, float *weight5pcf);

void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3);
