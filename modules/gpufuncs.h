typedef std::complex<double> Complex;

void gpu_add_to_power5(double *fivepcf, double *weight5pcf, Complex* alm,
	Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
	int *lut5_l4, int *lut5_n,
	int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

void gpu_allocate_luts(int **p_lut5_l1, int **p_lut5_l2, int **p_lut5_l12, int **p_lut5_l3,
        int **p_lut5_l4, int **p_lut5_n,
        int **p_lut5_zeta, int **p_lut5_i, int **p_lut5_j, int **p_lut5_k, int **p_lut5_l,
        int nouter, int ninner);

void gpu_add_to_power5_orig(double *fivepcf, double *weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5);

void gpu_allocate_m_luts(int **p_lut5_m1, int **p_lut5_m2, int **p_lut5_m3, int nouter);

void gpu_allocate_fivepcf(double **p_fivepcf, double *fivepcf, int size);

void gpu_allocate_weight5pcf(double **p_weight5pcf, double *weight5pcf, int size);

void gpu_free_memory(double *fivepcf, double *weight5pcf,
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l);

void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3);
