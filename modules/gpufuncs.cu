#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include "gpufuncs.h"
#include <thrust/complex.h>
#include <cuComplex.h>

int count = 0;
int pstart = 0; //particle number used for indexing
int pstart5 = 0;
int pstart3 = 0;
int pstart_discon = 0;
int pstart_discon2 = 0;
int pstart6 = 0;
thrust::complex<double>* d_alm, *d_almconj; //define d_alm and d_almconj here
thrust::complex<float>* f_alm, *f_almconj; //for use in float kernels

// ALM COMPUTATION (in beta)

__global__ void accumulate_multipoles_kernel(double *mult, int *mult_ct, double *x_array, double *y_array, double *z_array, double *w_array, int *bin_array, int length, int nmult, int order){
	// Accumulate the powers of x^p y^q z^r into the mult array

	//thread index i
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= length) return;

	// Compute x,y,z
	double x = x_array[i];
	double y = y_array[i];
	double z = z_array[i];
	double w = w_array[i];
	int bin = bin_array[i];

	double fi, fij, fijk;
	int count = bin*nmult;

	mult_ct[bin]++;

	fi = w;
	for(int i=0;i<=order;i++) {
			fij = fi;
			for(int j=0;j<=order-i;j++) {
					fijk = fij;
					for(int k=0;k<=order-i-j;k++) {
							mult[count++] += fijk;
							fijk *= z;
							}
					fij *= y;
					}
			fi *= x;
		}
}

void copy_mult(double **p_mult, double *mult, int **p_mult_ct, int *mult_ct, int size, int size_ct){
	// copy back to host
	cudaMemcpy(mult, (*p_mult), size*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(mult_ct, (*p_mult_ct), size_ct*sizeof(int), cudaMemcpyDeviceToHost);
}

void gpu_free_mult(double *mult, int *mult_ct){
	// free memory
	cudaFree(mult);
	cudaFree(mult_ct);
}

void gpu_allocate_mult(double **p_mult, double *mult, int **p_mult_ct, int *mult_ct, int size, int size_ct) {
  cudaMallocManaged(&(*p_mult), size*sizeof(double));
	double *d_mult = *(p_mult);
	for (int i = 0; i < size; i++) d_mult[i] = mult[i];
	cudaMallocManaged(&(*p_mult_ct), size_ct*sizeof(int));
	int *d_mult_ct = *(p_mult_ct);
	for (int i = 0; i < size_ct; i++) d_mult_ct[i] = mult_ct[i];
}

void accumulate_multipoles(double *d_mult, int *d_mult_ct, double *x_array, double *y_array, double *z_array, double *w_array, int *bin_array, int length, int max_length, int nmult, int order) {

	// array allocations for device
	double *dx_array, *dy_array, *dz_array, *dw_array;
	int *dbin_array;
  cudaMalloc(&dx_array, max_length*sizeof(double));
	cudaMalloc(&dy_array, max_length*sizeof(double));
	cudaMalloc(&dz_array, max_length*sizeof(double));
	cudaMalloc(&dw_array, max_length*sizeof(double));
	cudaMalloc(&dbin_array, max_length*sizeof(int));

	cudaMemcpy(dx_array, x_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dy_array, y_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dz_array, z_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dw_array, w_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dbin_array, bin_array, max_length*sizeof(int), cudaMemcpyHostToDevice);

  // Invoke kernel
  long threads = length;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

	accumulate_multipoles_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_mult,d_mult_ct,dx_array,dy_array,dz_array,dw_array,dbin_array,length,nmult,order);

	// Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(dx_array);
	cudaFree(dy_array);
	cudaFree(dz_array);
	cudaFree(dw_array);
  cudaFree(dbin_array);
}

// ======================================================= /
//  GPU KERNELS                                            /
// ======================================================= /


// ======================================================= /
//  ALL ADD_TO_POWER FUNCTIONS ARE HERE                    /
// ======================================================= /

//3PCF kernels
//Only 1 kernel option for 3CF, 3 precision modes
__global__ void add_to_power3_kernel_orig(double *threepcf, double *weight3pcf,
        double *weights, thrust::complex<double>* alm, thrust::complex<double> *almconj,
        int *lut3_i, int *lut3_j, int *lut3_ct, int nb, int nlm,
	int nouter, int order, int np, int pstart) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= nouter * np) return;
    //compute indices for LUTs
    int ip = i/nouter; //particle number
    int iouter = i%nouter; //index in LUTs
    //outer loop indices
    int ii = lut3_i[iouter];
    int jj = lut3_j[iouter];
    int ct = lut3_ct[iouter];
    int almidx = ip*nb*nlm;
    //calc weight
    double wp = weights[ip+pstart];
    //calc indices outside of loop
    int idx1 = almidx+ii*nlm;
    int idx2 = almidx+jj*nlm;

    for (int ell=0, n=0; ell<=order; ell++) {
      for (int mm=0; mm<=ell; mm++, n++) {
        atomicAdd(&threepcf[ell*nouter+ct], wp*(alm[idx1+n]*almconj[idx2+n]).real()*weight3pcf[n]);
      }
    }
}

__global__ void add_to_power3_kernel_orig_float(float *threepcf, float *weight3pcf,
        double *weights, thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut3_i, int *lut3_j, int *lut3_ct, int nb, int nlm,
        int nouter, int order, int np, int pstart) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= nouter * np) return;
    //compute indices for LUTs
    int ip = i/nouter; //particle number
    int iouter = i%nouter; //index in LUTs
    //outer loop indices
    int ii = lut3_i[iouter];
    int jj = lut3_j[iouter];
    int ct = lut3_ct[iouter];
    int almidx = ip*nb*nlm;
    //calc weight
    float wp = (float)weights[ip+pstart];
    //calc indices outside of loop
    int idx1 = almidx+ii*nlm;
    int idx2 = almidx+jj*nlm;

    for (int ell=0, n=0; ell<=order; ell++) {
      for (int mm=0; mm<=ell; mm++, n++) {
        atomicAdd(&threepcf[ell*nouter+ct], wp*(alm[idx1+n]*almconj[idx2+n]).real()*weight3pcf[n]);
      }
    }
}

__global__ void add_to_power3_kernel_orig_mixed(double *threepcf, double *weight3pcf,
        double *weights, thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut3_i, int *lut3_j, int *lut3_ct, int nb, int nlm,
        int nouter, int order, int np, int pstart) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= nouter * np) return;
    //compute indices for LUTs
    int ip = i/nouter; //particle number
    int iouter = i%nouter; //index in LUTs
    //outer loop indices
    int ii = lut3_i[iouter];
    int jj = lut3_j[iouter];
    int ct = lut3_ct[iouter]; 
    int almidx = ip*nb*nlm;
    //calc weight
    float wp = (float)weights[ip+pstart];
    //calc indices outside of loop
    int idx1 = almidx+ii*nlm;
    int idx2 = almidx+jj*nlm;

    for (int ell=0, n=0; ell<=order; ell++) {
      for (int mm=0; mm<=ell; mm++, n++) {
        atomicAdd(&threepcf[ell*nouter+ct], wp*(alm[idx1+n]*almconj[idx2+n]).real()*weight3pcf[n]);
      }
    }
}


// ======================================================= /

//DISCONNECTED 4PCF kernels
//Only 1 kernel option for DISCONNECTED 4PCF, 3 precision modes
__global__ void add_to_power_discon1_kernel_orig(double *discon1_r,
	double *discon1_i, double *weightdiscon,
	double *weights, thrust::complex<double>* alm,
	thrust::complex<double> *almconj, int *lut_discon_ell,
	int *lut_discon_mm, int nb, int nlm, int ndiscon, int order, int np,
	int pstart) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= ndiscon * np) return;
    //compute indices for LUTs
    int ip = i/ndiscon; //particle number
    int idx = i%ndiscon; //index in discon1
    int n = idx / nb; //index in LUTs (n) 
    int binidx = idx % nb; //bin idx (i)
    int almidx = ip*nb*nlm + binidx*nlm;
    //outer loop indices
    int ell = lut_discon_ell[n];
    int mm = lut_discon_mm[n];
    //calc weight
    double wp = weights[ip+pstart];
    //calc indices outside of loop
    double weight1 = wp*weightdiscon[n];

    if (mm < 0) {
      thrust::complex<double> delta = almconj[almidx+ell*(ell+1)/2-mm];
      atomicAdd(&discon1_r[n*nb+binidx], weight1*(delta.real()));
      atomicAdd(&discon1_i[n*nb+binidx], weight1*(delta.imag()));
    } else {
      thrust::complex<double> delta = alm[almidx+ell*(ell+1)/2+mm];
      atomicAdd(&discon1_r[n*nb+binidx], weight1*(delta.real()));
      atomicAdd(&discon1_i[n*nb+binidx], weight1*(delta.imag()));
    }
}

__global__ void add_to_power_discon1_kernel_orig_float(float *discon1_r,
	float *discon1_i, float *weightdiscon,
        double *weights, thrust::complex<float>* alm,
        thrust::complex<float> *almconj, int *lut_discon_ell,
        int *lut_discon_mm, int nb, int nlm, int ndiscon, int order, int np,
        int pstart) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= ndiscon * np) return;
    //compute indices for LUTs
    int ip = i/ndiscon; //particle number
    int idx = i%ndiscon; //index in discon1
    int n = idx / nb; //index in LUTs (n) 
    int binidx = idx % nb; //bin idx (i)
    int almidx = ip*nb*nlm + binidx*nlm;
    //outer loop indices
    int ell = lut_discon_ell[n];
    int mm = lut_discon_mm[n];
    //calc weight
    float wp = (float)weights[ip+pstart];
    //calc indices outside of loop
    float weight1 = wp*weightdiscon[n];
    if (mm < 0) {
      thrust::complex<float> delta = almconj[almidx+ell*(ell+1)/2-mm];
      atomicAdd(&discon1_r[n*nb+binidx], weight1*(delta.real()));
      atomicAdd(&discon1_i[n*nb+binidx], weight1*(delta.imag()));
    } else {
      thrust::complex<float> delta = alm[almidx+ell*(ell+1)/2+mm];
      atomicAdd(&discon1_r[n*nb+binidx], weight1*(delta.real()));
      atomicAdd(&discon1_i[n*nb+binidx], weight1*(delta.imag()));
    }
}

__global__ void add_to_power_discon1_kernel_orig_mixed(double *discon1_r,
	double *discon1_i, double *weightdiscon,
        double *weights, thrust::complex<float>* alm,
        thrust::complex<float> *almconj, int *lut_discon_ell,
        int *lut_discon_mm, int nb, int nlm, int ndiscon, int order, int np,
        int pstart) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= ndiscon * np) return;
    //compute indices for LUTs
    int ip = i/ndiscon; //particle number
    int idx = i%ndiscon; //index in discon1
    int n = idx / nb; //index in LUTs (n) 
    int binidx = idx % nb; //bin idx (i)
    int almidx = ip*nb*nlm + binidx*nlm;
    //outer loop indices
    int ell = lut_discon_ell[n];
    int mm = lut_discon_mm[n];
    //calc weight
    float wp = (float)weights[ip+pstart];
    //calc indices outside of loop
    double weight1 = wp*weightdiscon[n];
    if (mm < 0) {
      thrust::complex<float> delta = almconj[almidx+ell*(ell+1)/2-mm];
      atomicAdd(&discon1_r[n*nb+binidx], weight1*(delta.real()));
      atomicAdd(&discon1_i[n*nb+binidx], weight1*(delta.imag()));
    } else {
      thrust::complex<float> delta = alm[almidx+ell*(ell+1)/2+mm];
      atomicAdd(&discon1_r[n*nb+binidx], weight1*(delta.real()));
      atomicAdd(&discon1_i[n*nb+binidx], weight1*(delta.imag()));
    }
}

// ======================================================= /

//DISCON2 term
__global__ void add_to_power_discon2_kernel_orig(double *discon2_r,
        double *discon2_i, double *weightdiscon,
        double *weights, thrust::complex<double>* alm,
        thrust::complex<double> *almconj, int *lut_discon_ell1,
	int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
	int nb, int nlm, int nouter, int order, int ninner, int np,
        int pstart) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= nouter * np) return;
    //compute indices for LUTs
    int ip = i/nouter; //particle number
    int ct_ang = i % nouter; //ct_ang = 0 to 1295 for nl=6
    int nl2 = (order+1)*(order+1); // 36
    int n1 = ct_ang / nl2; //n1 outer index
    int n2 = ct_ang % nl2; //n2 inner index

    int almidx = ip*nb*nlm;
    //outer loop indices
    int ell1 = lut_discon_ell1[ct_ang];
    int mm1 = lut_discon_mm1[ct_ang];
    int ell2 = lut_discon_ell2[ct_ang];
    int mm2 = lut_discon_mm2[ct_ang];
    //calc weight
    double wp = weights[ip+pstart];
    //calc indices outside of loop
    double weight1 = wp*weightdiscon[n1];
    double weight2 = weight1*weightdiscon[n2];

    if (mm1 < 0) {
      for (int ii = 0, ct_rad = 0; ii < nb; ii++) {
	thrust::complex<double> alm1 = weight2*almconj[almidx+ii*nlm+ell1*(ell1+1)/2-mm1];
        if (mm2 < 0) {
          for (int jj = ii+1; jj < nb; jj++, ct_rad++) {
            thrust::complex<double> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
            atomicAdd(&discon2_r[ct_ang*ninner+ct_rad], delta.real());
            atomicAdd(&discon2_i[ct_ang*ninner+ct_rad], delta.imag());
          }
        } else {
          for (int jj = ii+1; jj < nb; jj++, ct_rad++) {
            thrust::complex<double> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
            atomicAdd(&discon2_r[ct_ang*ninner+ct_rad], delta.real());
            atomicAdd(&discon2_i[ct_ang*ninner+ct_rad], delta.imag());
          }
        }
      }
    } else {
      for (int ii = 0, ct_rad = 0; ii < nb; ii++) {
        thrust::complex<double> alm1 = weight2*alm[almidx+ii*nlm+ell1*(ell1+1)/2+mm1];
        if (mm2 < 0) {
          for (int jj = ii+1; jj < nb; jj++, ct_rad++) {
            thrust::complex<double> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
            atomicAdd(&discon2_r[ct_ang*ninner+ct_rad], delta.real());
            atomicAdd(&discon2_i[ct_ang*ninner+ct_rad], delta.imag());
          }
        } else {
          for (int jj = ii+1; jj < nb; jj++, ct_rad++) {
            thrust::complex<double> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
            atomicAdd(&discon2_r[ct_ang*ninner+ct_rad], delta.real());
            atomicAdd(&discon2_i[ct_ang*ninner+ct_rad], delta.imag());
          }
        }
      }
    }
}

__global__ void add_to_power_discon2_kernel_b(double *discon2_r,
	double *discon2_i, double *weightdiscon, double wp,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut_discon_ell1, int *lut_discon_ell2, int *lut_discon_mm1,
	int *lut_discon_mm2, int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner, int almidx) {
    //wp and almidx passed as scalars 
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int ct_ang = i % nouter; //ct_ang = 0 to 1295 for nl=6
    int iinner = i / nouter; //0 to 189
    int ct_rad = iinner;
    int nl2 = (order+1)*(order+1); // 36
    int n1 = ct_ang / nl2; //n1 outer index
    int n2 = ct_ang % nl2; //n2 inner index

    //outer loop indices
    int ell1 = lut_discon_ell1[ct_ang];
    int mm1 = lut_discon_mm1[ct_ang];
    int ell2 = lut_discon_ell2[ct_ang];
    int mm2 = lut_discon_mm2[ct_ang];
    //calc indices outside of loop
    double weight1 = wp*weightdiscon[n1];
    double weight2 = weight1*weightdiscon[n2];

    int ii = lut_discon_i[iinner];
    int jj = lut_discon_j[iinner];

    if (mm1 < 0) {
      thrust::complex<double> alm1 = weight2*almconj[almidx+ii*nlm+ell1*(ell1+1)/2-mm1];
      if (mm2 < 0) {
        thrust::complex<double> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
        discon2_r[ct_ang*ninner+ct_rad] += delta.real();
        discon2_i[ct_ang*ninner+ct_rad] += delta.imag();
      } else {
        thrust::complex<double> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
        discon2_r[ct_ang*ninner+ct_rad] += delta.real();
        discon2_i[ct_ang*ninner+ct_rad] += delta.imag();
      }
    } else {
      thrust::complex<double> alm1 = weight2*alm[almidx+ii*nlm+ell1*(ell1+1)/2+mm1];
      if (mm2 < 0) {
        thrust::complex<double> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
        discon2_r[ct_ang*ninner+ct_rad] += delta.real();
        discon2_i[ct_ang*ninner+ct_rad] += delta.imag();
      } else {
        thrust::complex<double> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
        discon2_r[ct_ang*ninner+ct_rad] += delta.real();
        discon2_i[ct_ang*ninner+ct_rad] += delta.imag();
      }
    }
}

__global__ void add_to_power_discon2_kernel_final(double *discon2_r,
	double *discon2_i, double *weightdiscon, double *weights,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut_discon_ell1, int *lut_discon_ell2, int *lut_discon_mm1,
	int *lut_discon_mm2, int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner,
	int np, int nprnd, int npblocks, int pstart, int qbalance, int qinvert) {
    //wp and almidx passed as scalars 
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= npblocks * nouter * ninner) return;
    //compute indices for LUTs
    int ip0 = i/(nouter*ninner)*DISCON2_PARTICLES_PER_THREAD; //START particle number
    int ipend = ip0+DISCON2_PARTICLES_PER_THREAD;
    if (ipend > np) ipend = np;
    int didx = i % (nouter*ninner); //index in discon2

    int ct_ang = didx % nouter; //ct_ang = 0 to 1295 for nl=6
    int iinner = didx / nouter; //0 to 189
    int ct_rad = iinner;

    int nl2 = (order+1)*(order+1); // 36
    int n1 = ct_ang / nl2; //n1 outer index
    int n2 = ct_ang % nl2; //n2 inner index

    //outer loop indices
    int ell1 = lut_discon_ell1[ct_ang];
    int mm1 = lut_discon_mm1[ct_ang];
    int ell2 = lut_discon_ell2[ct_ang];
    int mm2 = lut_discon_mm2[ct_ang];

    int ii = lut_discon_i[iinner];
    int jj = lut_discon_j[iinner];

    //local register to hold sum for this 1000 particles
    double d2_r = 0, d2_i = 0;

    for (int jp = ip0; jp < ipend; jp++) {
      //calc weight
      double wp = weights[jp+pstart];

      if (!(((wp<0)&&(qbalance))||(qinvert))) continue;

      //calc indices inside of loop
      double weight1 = wp*weightdiscon[n1];
      double weight2 = weight1*weightdiscon[n2];
      int almidx = jp*nb*nlm;
      if (mm1 < 0) {
        thrust::complex<double> alm1 = weight2*almconj[almidx+ii*nlm+ell1*(ell1+1)/2-mm1];
        if (mm2 < 0) {
          thrust::complex<double> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        } else {
          thrust::complex<double> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        }
      } else {
        thrust::complex<double> alm1 = weight2*alm[almidx+ii*nlm+ell1*(ell1+1)/2+mm1];
        if (mm2 < 0) {
          thrust::complex<double> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        } else {
          thrust::complex<double> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        }
      }
    }
    //now atomicAdd
    atomicAdd(&discon2_r[ct_ang*ninner+ct_rad], d2_r);
    atomicAdd(&discon2_i[ct_ang*ninner+ct_rad], d2_i);
}

__global__ void add_to_power_discon2_kernel_final_float(float *discon2_r,
	float *discon2_i, float *weightdiscon, double *weights,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut_discon_ell1, int *lut_discon_ell2, int *lut_discon_mm1,
	int *lut_discon_mm2, int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner,
	int np, int nprnd, int npblocks, int pstart, int qbalance, int qinvert) {
    //wp and almidx passed as scalars 
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= npblocks * nouter * ninner) return;
    //compute indices for LUTs
    int ip0 = i/(nouter*ninner)*DISCON2_PARTICLES_PER_THREAD; //START particle number
    int ipend = ip0+DISCON2_PARTICLES_PER_THREAD;
    if (ipend > np) ipend = np;
    int didx = i % (nouter*ninner); //index in discon2

    int ct_ang = didx % nouter; //ct_ang = 0 to 1295 for nl=6
    int iinner = didx / nouter; //0 to 189
    int ct_rad = iinner;

    int nl2 = (order+1)*(order+1); // 36
    int n1 = ct_ang / nl2; //n1 outer index
    int n2 = ct_ang % nl2; //n2 inner index

    //outer loop indices
    int ell1 = lut_discon_ell1[ct_ang];
    int mm1 = lut_discon_mm1[ct_ang];
    int ell2 = lut_discon_ell2[ct_ang];
    int mm2 = lut_discon_mm2[ct_ang];

    int ii = lut_discon_i[iinner];
    int jj = lut_discon_j[iinner];

    //local register to hold sum for this 1000 particles
    float d2_r = 0, d2_i = 0;

    for (int jp = ip0; jp < ipend; jp++) {
      //calc weight
      float wp = (float)weights[jp+pstart];

      if (!(((wp<0)&&(qbalance))||(qinvert))) continue;

      //calc indices inside of loop
      float weight1 = wp*weightdiscon[n1];
      float weight2 = weight1*weightdiscon[n2];
      int almidx = jp*nb*nlm;
      if (mm1 < 0) {
        thrust::complex<float> alm1 = weight2*almconj[almidx+ii*nlm+ell1*(ell1+1)/2-mm1];
        if (mm2 < 0) {
          thrust::complex<float> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        } else {
          thrust::complex<float> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        }
      } else {
        thrust::complex<float> alm1 = weight2*alm[almidx+ii*nlm+ell1*(ell1+1)/2+mm1];
        if (mm2 < 0) {
          thrust::complex<float> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        } else {
          thrust::complex<float> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        }
      }
    }
    //now atomicAdd
    atomicAdd(&discon2_r[ct_ang*ninner+ct_rad], d2_r);
    atomicAdd(&discon2_i[ct_ang*ninner+ct_rad], d2_i);
}

__global__ void add_to_power_discon2_kernel_final_mixed(double *discon2_r,
	double *discon2_i, double *weightdiscon, double *weights,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut_discon_ell1, int *lut_discon_ell2, int *lut_discon_mm1,
	int *lut_discon_mm2, int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner,
	int np, int nprnd, int npblocks, int pstart, int qbalance, int qinvert) {
    //wp and almidx passed as scalars 
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //nouter = dim of LUTs = N3PCF
    if (i >= npblocks * nouter * ninner) return;
    //compute indices for LUTs
    int ip0 = i/(nouter*ninner)*DISCON2_PARTICLES_PER_THREAD; //START particle number
    int ipend = ip0+DISCON2_PARTICLES_PER_THREAD;
    if (ipend > np) ipend = np;
    int didx = i % (nouter*ninner); //index in discon2

    int ct_ang = didx % nouter; //ct_ang = 0 to 1295 for nl=6
    int iinner = didx / nouter; //0 to 189
    int ct_rad = iinner;

    int nl2 = (order+1)*(order+1); // 36
    int n1 = ct_ang / nl2; //n1 outer index
    int n2 = ct_ang % nl2; //n2 inner index

    //outer loop indices
    int ell1 = lut_discon_ell1[ct_ang];
    int mm1 = lut_discon_mm1[ct_ang];
    int ell2 = lut_discon_ell2[ct_ang];
    int mm2 = lut_discon_mm2[ct_ang];

    int ii = lut_discon_i[iinner];
    int jj = lut_discon_j[iinner];

    //local register to hold sum for this 1000 particles
    double d2_r = 0, d2_i = 0;

    for (int jp = ip0; jp < ipend; jp++) {
      //calc weight
      float wp = (float)weights[jp+pstart];

      if (!(((wp<0)&&(qbalance))||(qinvert))) continue;

      //calc indices inside of loop
      float weight1 = wp*weightdiscon[n1];
      float weight2 = weight1*weightdiscon[n2];
      int almidx = jp*nb*nlm;
      if (mm1 < 0) {
        thrust::complex<float> alm1 = weight2*almconj[almidx+ii*nlm+ell1*(ell1+1)/2-mm1];
        if (mm2 < 0) {
          thrust::complex<float> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        } else {
          thrust::complex<float> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        }
      } else {
        thrust::complex<float> alm1 = weight2*alm[almidx+ii*nlm+ell1*(ell1+1)/2+mm1];
        if (mm2 < 0) {
          thrust::complex<float> delta = alm1*almconj[almidx+jj*nlm+ell2*(ell2+1)/2-mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        } else {
          thrust::complex<float> delta = alm1*alm[almidx+jj*nlm+ell2*(ell2+1)/2+mm2];
          d2_r += delta.real();
          d2_i += delta.imag();
        }
      }
    }
    //now atomicAdd
    atomicAdd(&discon2_r[ct_ang*ninner+ct_rad], d2_r);
    atomicAdd(&discon2_i[ct_ang*ninner+ct_rad], d2_i);
}

// ======================================================= /


//4PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes

__global__ void add_to_power4_kernel(double *fourpcf, double *weight4pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd, int *lut4_n,
	int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k, 
        double wp, int nlm, int nouter, int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut4_zeta[iouter]+iinner;
    double pcf_element = fourpcf[bin_index]; // this element
    //cald weight
    double weight = weight4pcf[lut4_n[iouter]];
    //outer loop indices
    int l1 = lut4_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut4_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut4_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    bool odd = lut4_odd[iouter]; // this defines whether we have an odd-parity multiplet
    int n = lut4_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut4_i[iinner];
    int j = lut4_j[iinner];
    int k = lut4_k[iinner];
    //alms
    thrust::complex<double> alm1w = 0;
    thrust::complex<double> alm2 = 0;
    int m3, tmp_lm3;
    double delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          m3 = -m1-m2;
          if (m3<0) continue; // only need to use m3>=0
          if (m3>l3) continue; // this violates triangle conditions

          // Look up the relevant weight
          weight = weight4pcf[n++];
          if (weight==0) continue;
          tmp_lm3 = tmp_l3+m3;
          // Create temporary copy of a_l2m2 and a_l3m3, taking conjugate if necessary
          // No conjugates needed for a_l3m3 since we fixed m3>=0!
          // Note we add the coupling weight factor to a_l3m3
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
        
	  //calculate delta
          delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).imag(); //odd parity
	  //add to this element
	  pcf_element += delta;
        }
      }
    } else {
        for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          m3 = -m1-m2;
          if (m3<0) continue; // only need to use m3>=0
          if (m3>l3) continue; // this violates triangle conditions

          // Look up the relevant weight
          weight = weight4pcf[n++];
          if (weight==0) continue;
          tmp_lm3 = tmp_l3+m3;
          // Create temporary copy of a_l2m2 and a_l3m3, taking conjugate if necessary
          // No conjugates needed for a_l3m3 since we fixed m3>=0!
          // Note we add the coupling weight factor to a_l3m3
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];

          //calculate delta
          //if (odd) delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).imag(); //odd parity
          //else delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real(); //even parity
          delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real();
          //add to this element
          pcf_element += delta;
        }
      }
    }
    fourpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power4_kernel_float(float *fourpcf, float *weight4pcf,
        thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd, int *lut4_n,
        int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nlm, int nouter, int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut4_zeta[iouter]+iinner;
    float pcf_element = fourpcf[bin_index]; // this element
    //cald weight
    float weight = weight4pcf[lut4_n[iouter]];
    //outer loop indices
    int l1 = lut4_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut4_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut4_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    bool odd = lut4_odd[iouter]; // this defines whether we have an odd-parity multiplet
    int n = lut4_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut4_i[iinner];
    int j = lut4_j[iinner];
    int k = lut4_k[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    int m3, tmp_lm3;
    float delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          m3 = -m1-m2;
          if (m3<0) continue; // only need to use m3>=0
          if (m3>l3) continue; // this violates triangle conditions

          // Look up the relevant weight
          weight = weight4pcf[n++];
          if (weight==0) continue;
          tmp_lm3 = tmp_l3+m3;
          // Create temporary copy of a_l2m2 and a_l3m3, taking conjugate if necessary
          // No conjugates needed for a_l3m3 since we fixed m3>=0!
          // Note we add the coupling weight factor to a_l3m3
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];

          //calculate delta
          delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).imag(); //odd parity
          //add to this element
          pcf_element += delta;
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          m3 = -m1-m2;
          if (m3<0) continue; // only need to use m3>=0
          if (m3>l3) continue; // this violates triangle conditions

          // Look up the relevant weight
          weight = weight4pcf[n++];
          if (weight==0) continue;
          tmp_lm3 = tmp_l3+m3;
          // Create temporary copy of a_l2m2 and a_l3m3, taking conjugate if necessary
          // No conjugates needed for a_l3m3 since we fixed m3>=0!
          // Note we add the coupling weight factor to a_l3m3
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];

          //calculate delta
          delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real(); //even parity
          //add to this element
          pcf_element += delta;
        }
      }
    }
    fourpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power4_kernel_mixed(double *fourpcf, double *weight4pcf,
        thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd, int *lut4_n,
        int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nlm, int nouter, int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut4_zeta[iouter]+iinner;
    double pcf_element = fourpcf[bin_index]; // this element
    //cald weight
    double weight = weight4pcf[lut4_n[iouter]];
    //outer loop indices
    int l1 = lut4_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut4_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut4_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    bool odd = lut4_odd[iouter]; // this defines whether we have an odd-parity multiplet
    int n = lut4_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut4_i[iinner];
    int j = lut4_j[iinner];
    int k = lut4_k[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    int m3, tmp_lm3;
    double delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          m3 = -m1-m2;
          if (m3<0) continue; // only need to use m3>=0
          if (m3>l3) continue; // this violates triangle conditions

          // Look up the relevant weight
          weight = weight4pcf[n++];
          if (weight==0) continue;
          tmp_lm3 = tmp_l3+m3;
          // Create temporary copy of a_l2m2 and a_l3m3, taking conjugate if necessary
          // No conjugates needed for a_l3m3 since we fixed m3>=0!
          // Note we add the coupling weight factor to a_l3m3
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];

          //calculate delta
          delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).imag(); //odd parity
          //add to this element
          pcf_element += delta;
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          m3 = -m1-m2;
          if (m3<0) continue; // only need to use m3>=0
          if (m3>l3) continue; // this violates triangle conditions

          // Look up the relevant weight
          weight = weight4pcf[n++];
          if (weight==0) continue;
          tmp_lm3 = tmp_l3+m3;
          // Create temporary copy of a_l2m2 and a_l3m3, taking conjugate if necessary
          // No conjugates needed for a_l3m3 since we fixed m3>=0!
          // Note we add the coupling weight factor to a_l3m3
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];

          //calculate delta
          delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real(); //even parity
          //add to this element
          pcf_element += delta;
        }
      }
    }
    fourpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power4_kernel_orig(double *fourpcf, double *weight4pcf,
        thrust::complex<double>* alm, thrust::complex<double> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd, int *lut4_m1,
	int *lut4_m2, int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j,
	int *lut4_k, double wp, int nlm, int nouter, int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut4_zeta[iouter]+iinner;
    //outer loop indices
    int l1 = lut4_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut4_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut4_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    bool odd = lut4_odd[iouter]; // this defines whether we have an odd-parity multiplet
    int m1 = lut4_m1[iouter];
    int m2 = lut4_m2[iouter];
    int n = lut4_n[iouter]; 
    //calc weight
    double weight = weight4pcf[n];
    //inner loop indices
    int ii = lut4_i[iinner];
    int j = lut4_j[iinner];
    int k = lut4_k[iinner];
    //alms
    thrust::complex<double> alm1w = 0;
    thrust::complex<double> alm2 = 0;
    //alm idx = particle_num * nbin * nlm
    if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
    int m3 = -m1-m2;
    int tmp_lm3 = tmp_l3+m3;
    //double delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real(); //even parity
    //atomicAdd(&fourpcf[bin_index], delta);
    if (odd) atomicAdd(&fourpcf[bin_index],weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).imag()); else atomicAdd(&fourpcf[bin_index],weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real());
    //atomicAdd(&fourpcf[bin_index], 1);
}

__global__ void add_to_power4_kernel_orig_float(float *fourpcf,
	float *weight4pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut4_l1, int *lut4_l2,
	int *lut4_l3, bool *lut4_odd, int *lut4_m1, int *lut4_m2, int *lut4_n,
	int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k, float wp, 
	int nlm, int nouter, int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut4_zeta[iouter]+iinner;
    //outer loop indices
    int l1 = lut4_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut4_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut4_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    bool odd = lut4_odd[iouter]; // this defines whether we have an odd-parity multiplet
    int m1 = lut4_m1[iouter];
    int m2 = lut4_m2[iouter];
    int n = lut4_n[iouter]; 
    //calc weight
    float weight = weight4pcf[n];
    //inner loop indices
    int ii = lut4_i[iinner];
    int j = lut4_j[iinner];
    int k = lut4_k[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
    int m3 = -m1-m2;
    int tmp_lm3 = tmp_l3+m3;
    //float delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real(); //even parity
    //atomicAdd(&fourpcf[bin_index], delta);
    if (odd) atomicAdd(&fourpcf[bin_index],weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).imag()); else atomicAdd(&fourpcf[bin_index],weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real());
}

__global__ void add_to_power4_kernel_orig_mixed(double *fourpcf,
	double *weight4pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut4_l1, int *lut4_l2,
	int *lut4_l3, bool *lut4_odd, int *lut4_m1, int *lut4_m2, int *lut4_n,
	int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k, float wp, 
	int nlm, int nouter, int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut4_zeta[iouter]+iinner;
    //outer loop indices
    int l1 = lut4_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut4_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut4_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    bool odd = lut4_odd[iouter]; // this defines whether we have an odd-parity multiplet
    int m1 = lut4_m1[iouter];
    int m2 = lut4_m2[iouter];
    int n = lut4_n[iouter]; 
    //calc weight
    double weight = weight4pcf[n];
    //inner loop indices
    int ii = lut4_i[iinner];
    int j = lut4_j[iinner];
    int k = lut4_k[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
    int m3 = -m1-m2;
    int tmp_lm3 = tmp_l3+m3;
    //double delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real(); //even parity
    //atomicAdd(&fourpcf[bin_index], delta);
    if (odd) atomicAdd(&fourpcf[bin_index],weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).imag()); else atomicAdd(&fourpcf[bin_index],weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real());
}

// ======================================================= /

//5PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes

__global__ void add_to_power5_kernel(double *fivepcf, double *weight5pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j,
	int *lut5_k, int *lut5_l, double wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut5_zeta[iouter]+iinner;
    double pcf_element = fivepcf[bin_index]; // this element
    //cald weight
    double weight = weight5pcf[lut5_n[iouter]];
    //outer loop indices
    int l1 = lut5_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut5_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l12 = lut5_l12[iouter];
    int l3 = lut5_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut5_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    bool odd = lut5_odd[iouter];
    int n = lut5_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut5_i[iinner];
    int j = lut5_j[iinner];
    int k = lut5_k[iinner];
    int l = lut5_l[iinner];
    //alms
    thrust::complex<double> alm1w = 0;
    thrust::complex<double> alm2 = 0;
    thrust::complex<double> alm3 = 0;
    int m4, tmp_lm4;
    double delta;

    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            m4 = -m1-m2-m3;
            if (m4<0) continue; // only need to use m4>=0
            if (m4>l4) continue; // this violates triangle conditions
            // Look up the relevant weight
            weight = weight5pcf[n++];
            if (weight==0) continue;
            tmp_lm4 = tmp_l4+m4;
            // Create temporary copies of a_l3m3 and a_l4m4, taking conjugates if necessary
            // No conjugates needed for a_l4m4 since we fixed m4>=0!
            // Note we add the coupling weight factor to a_l4m4
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            //calculate delta
            delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).imag(); //odd parity
            //add to this element
            pcf_element += delta;
          }
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            m4 = -m1-m2-m3;
            if (m4<0) continue; // only need to use m4>=0
            if (m4>l4) continue; // this violates triangle conditions
            // Look up the relevant weight
            weight = weight5pcf[n++];
            if (weight==0) continue;
            tmp_lm4 = tmp_l4+m4;
            // Create temporary copies of a_l3m3 and a_l4m4, taking conjugates if necessary
            // No conjugates needed for a_l4m4 since we fixed m4>=0!
            // Note we add the coupling weight factor to a_l4m4
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            //calculate delta
            delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real(); //even parity
            //add to this element
            pcf_element += delta;
          }
        }
      }
    }
    fivepcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power5_kernel_float(float *fivepcf, float *weight5pcf,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j,
	int *lut5_k, int *lut5_l, float wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut5_zeta[iouter]+iinner;
    float pcf_element = fivepcf[bin_index]; // this element
    //calc weight
    float weight = weight5pcf[lut5_n[iouter]];
    //outer loop indices
    int l1 = lut5_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut5_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l12 = lut5_l12[iouter];
    int l3 = lut5_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut5_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    bool odd = lut5_odd[iouter];
    int n = lut5_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut5_i[iinner];
    int j = lut5_j[iinner];
    int k = lut5_k[iinner];
    int l = lut5_l[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    thrust::complex<float> alm3 = 0;
    int m4, tmp_lm4;
    float delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            m4 = -m1-m2-m3;
            if (m4<0) continue; // only need to use m4>=0
            if (m4>l4) continue; // this violates triangle conditions
            // Look up the relevant weight
            weight = weight5pcf[n++];
            if (weight==0) continue;
            tmp_lm4 = tmp_l4+m4;
            // Create temporary copies of a_l3m3 and a_l4m4, taking conjugates if necessary
            // No conjugates needed for a_l4m4 since we fixed m4>=0!
            // Note we add the coupling weight factor to a_l4m4
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            //calculate delta
            delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).imag(); //odd parity
            //add to this element
            pcf_element += delta;
          }
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            m4 = -m1-m2-m3;
            if (m4<0) continue; // only need to use m4>=0
            if (m4>l4) continue; // this violates triangle conditions
            // Look up the relevant weight
            weight = weight5pcf[n++];
            if (weight==0) continue;
            tmp_lm4 = tmp_l4+m4;
            // Create temporary copies of a_l3m3 and a_l4m4, taking conjugates if necessary
            // No conjugates needed for a_l4m4 since we fixed m4>=0!
            // Note we add the coupling weight factor to a_l4m4
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            //calculate delta
            delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real(); //even parity
            //add to this element
            pcf_element += delta;
          }
        }
      }
    }
    fivepcf[bin_index] = pcf_element; 
}

__global__ void add_to_power5_kernel_mixed(double *fivepcf, double *weight5pcf,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j,
	int *lut5_k, int *lut5_l, float wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut5_zeta[iouter]+iinner;
    double pcf_element = fivepcf[bin_index]; // this element
    //calc weight
    double weight = weight5pcf[lut5_n[iouter]];
    //outer loop indices
    int l1 = lut5_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut5_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l12 = lut5_l12[iouter];
    int l3 = lut5_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut5_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    bool odd = lut5_odd[iouter];
    int n = lut5_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut5_i[iinner];
    int j = lut5_j[iinner];
    int k = lut5_k[iinner];
    int l = lut5_l[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    thrust::complex<float> alm3 = 0;
    int m4, tmp_lm4;
    double delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            m4 = -m1-m2-m3;
            if (m4<0) continue; // only need to use m4>=0
            if (m4>l4) continue; // this violates triangle conditions
            // Look up the relevant weight
            weight = weight5pcf[n++];
            if (weight==0) continue;
            tmp_lm4 = tmp_l4+m4;
            // Create temporary copies of a_l3m3 and a_l4m4, taking conjugates if necessary
            // No conjugates needed for a_l4m4 since we fixed m4>=0!
            // Note we add the coupling weight factor to a_l4m4
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            //calculate delta
            delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).imag(); //odd parity
            //add to this element
            pcf_element += delta;
          }
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            m4 = -m1-m2-m3;
            if (m4<0) continue; // only need to use m4>=0
            if (m4>l4) continue; // this violates triangle conditions
            // Look up the relevant weight
            weight = weight5pcf[n++];
            if (weight==0) continue;
            tmp_lm4 = tmp_l4+m4;
            // Create temporary copies of a_l3m3 and a_l4m4, taking conjugates if necessary
            // No conjugates needed for a_l4m4 since we fixed m4>=0!
            // Note we add the coupling weight factor to a_l4m4
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            //calculate delta
            delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real(); //even parity
            //add to this element
            pcf_element += delta;
          }
        }
      }
    }
    fivepcf[bin_index] = pcf_element; 
}

__global__ void add_to_power5_kernel_orig(double *fivepcf, double *weight5pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4, bool *lut5_odd,
	int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n, int *lut5_zeta,
	int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nlm, int nouter, int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut5_zeta[iouter]+iinner;
    //outer loop indices
    int l1 = lut5_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut5_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut5_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut5_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    bool odd = lut5_odd[iouter];
    int m1 = lut5_m1[iouter];
    int m2 = lut5_m2[iouter];
    int m3 = lut5_m3[iouter];
    int n = lut5_n[iouter]; //this is the starting n for this thread
    //calc weight
    double weight = weight5pcf[n];
    //inner loop indices
    int ii = lut5_i[iinner];
    int j = lut5_j[iinner];
    int k = lut5_k[iinner];
    int l = lut5_l[iinner];
    //alms
    thrust::complex<double> alm1w = 0;
    thrust::complex<double> alm2 = 0;
    thrust::complex<double> alm3 = 0;
    if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
    int m4 = -m1-m2-m3;
    int tmp_lm4 = tmp_l4+m4;
    if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
    //double delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real(); //even parity
    //atomicAdd(&fivepcf[bin_index], delta);
    if (odd) atomicAdd(&fivepcf[bin_index], weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).imag()); else atomicAdd(&fivepcf[bin_index], weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real()); 
}

__global__ void add_to_power5_kernel_orig_float(float *fivepcf,
	float *weight5pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2,
	int *lut5_l3, int *lut5_l4, bool *lut5_odd, int *lut5_m1, int *lut5_m2,
	int *lut5_m3, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j,
	int *lut5_k, int *lut5_l, float wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut5_zeta[iouter]+iinner;
    //outer loop indices
    int l1 = lut5_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut5_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut5_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut5_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    bool odd = lut5_odd[iouter];
    int m1 = lut5_m1[iouter];
    int m2 = lut5_m2[iouter];
    int m3 = lut5_m3[iouter];
    int n = lut5_n[iouter]; //this is the starting n for this thread
    //calc weight
    float weight = weight5pcf[n];
    //inner loop indices
    int ii = lut5_i[iinner];
    int j = lut5_j[iinner];
    int k = lut5_k[iinner];
    int l = lut5_l[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    thrust::complex<float> alm3 = 0;
    if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
    int m4 = -m1-m2-m3;
    int tmp_lm4 = tmp_l4+m4;
    if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
    //float delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real(); //even parity
    //atomicAdd(&fivepcf[bin_index], delta);
    if (odd) atomicAdd(&fivepcf[bin_index], weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).imag()); else atomicAdd(&fivepcf[bin_index], weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real());
}

__global__ void add_to_power5_kernel_orig_mixed(double *fivepcf,
	double *weight5pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2,
	int *lut5_l3, int *lut5_l4, bool *lut5_odd, int *lut5_m1, int *lut5_m2,
	int *lut5_m3, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j,
	int *lut5_k, int *lut5_l, float wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut5_zeta[iouter]+iinner;
    //outer loop indices
    int l1 = lut5_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut5_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l3 = lut5_l3[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut5_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    bool odd = lut5_odd[iouter];
    int m1 = lut5_m1[iouter];
    int m2 = lut5_m2[iouter];
    int m3 = lut5_m3[iouter];
    int n = lut5_n[iouter]; //this is the starting n for this thread
    //calc weight
    double weight = weight5pcf[n];
    //inner loop indices
    int ii = lut5_i[iinner];
    int j = lut5_j[iinner];
    int k = lut5_k[iinner];
    int l = lut5_l[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    thrust::complex<float> alm3 = 0;
    if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
    int m4 = -m1-m2-m3;
    int tmp_lm4 = tmp_l4+m4;
    if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
    //double delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real(); //even parity
    //atomicAdd(&fivepcf[bin_index], delta);
    if (odd) atomicAdd(&fivepcf[bin_index], weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).imag()); else atomicAdd(&fivepcf[bin_index], weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real());
}


// ======================================================= /

//6PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes

__global__ void add_to_power6_kernel(double *sixpcf, double *weight6pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
	int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
	int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j, int *lut6_k,
	int *lut6_l, int *lut6_m, double wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut6_zeta[iouter]+iinner;
    double pcf_element = sixpcf[bin_index]; // this element
    //cald weight
    double weight = weight6pcf[lut6_n[iouter]];
    //outer loop indices
    int l1 = lut6_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut6_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l12 = lut6_l12[iouter];
    int l3 = lut6_l3[iouter];
    int l123 = lut6_l123[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut6_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    int l5 = lut6_l5[iouter];
    int tmp_l5 = l5*(l5+1)/2;
    bool odd = lut6_odd[iouter];
    int n = lut6_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut6_i[iinner];
    int j = lut6_j[iinner];
    int k = lut6_k[iinner];
    int l = lut6_l[iinner];
    int m = lut6_m[iinner];
    //alms
    thrust::complex<double> alm1w = 0;
    thrust::complex<double> alm2 = 0;
    thrust::complex<double> alm3 = 0;
    thrust::complex<double> alm4 = 0;
    int m5, tmp_lm5;
    double delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
	    if(abs(m1+m2+m3)>l123) continue;
            // Create temporary copy of a_l3m3, taking conjugate if necessary
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            // Iterate over m4 (including negative)
            for(int m4=-l4; m4<=l4; m4++){
              m5 = -m1-m2-m3-m4;
              if (m5<0) continue; // only need to use m5>=0
              if (m5>l5) continue; // this violates triangle conditions
              // Look up the relevant weight
              weight = weight6pcf[n++];
              if (weight==0) continue;
              tmp_lm5 = tmp_l5+m5;
              // Create temporary copies of a_l4m4 and a_l5m5, taking conjugates if necessary
              // No conjugates needed for a_l5m5 since we fixed m5>=0!
              // Note we add the coupling weight factor to a_l5m5
              if (m4 < 0) alm4 = alm3*almconj[almidx+l*nlm+tmp_l4-m4]; else alm4 = alm3*alm[almidx+l*nlm+tmp_l4+m4];
              //calculate delta
              delta = weight*(alm4*alm[almidx+m*nlm+tmp_lm5]).imag(); //odd parity
              //add to this element
              pcf_element += delta;
            }
          }
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            if(abs(m1+m2+m3)>l123) continue;
            // Create temporary copy of a_l3m3, taking conjugate if necessary
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            // Iterate over m4 (including negative)
            for(int m4=-l4; m4<=l4; m4++){
              m5 = -m1-m2-m3-m4;
              if (m5<0) continue; // only need to use m5>=0
              if (m5>l5) continue; // this violates triangle conditions
              // Look up the relevant weight
              weight = weight6pcf[n++];
              if (weight==0) continue;
              tmp_lm5 = tmp_l5+m5;
              // Create temporary copies of a_l4m4 and a_l5m5, taking conjugates if necessary
              // No conjugates needed for a_l5m5 since we fixed m5>=0!
              // Note we add the coupling weight factor to a_l5m5
              if (m4 < 0) alm4 = alm3*almconj[almidx+l*nlm+tmp_l4-m4]; else alm4 = alm3*alm[almidx+l*nlm+tmp_l4+m4];
              //calculate delta
              delta = weight*(alm4*alm[almidx+m*nlm+tmp_lm5]).real(); //even parity
              //add to this element
              pcf_element += delta;
            }
          }
        }
      }
    }
    sixpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power6_kernel_float(float *sixpcf, float *weight6pcf,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
	int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
	int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j, int *lut6_k,
	int *lut6_l, int *lut6_m, float wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut6_zeta[iouter]+iinner;
    float pcf_element = sixpcf[bin_index]; // this element
    //cald weight
    float weight = weight6pcf[lut6_n[iouter]];
    //outer loop indices
    int l1 = lut6_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut6_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l12 = lut6_l12[iouter];
    int l3 = lut6_l3[iouter];
    int l123 = lut6_l123[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut6_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    int l5 = lut6_l5[iouter];
    int tmp_l5 = l5*(l5+1)/2;
    bool odd = lut6_odd[iouter];
    int n = lut6_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut6_i[iinner];
    int j = lut6_j[iinner];
    int k = lut6_k[iinner];
    int l = lut6_l[iinner];
    int m = lut6_m[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    thrust::complex<float> alm3 = 0;
    thrust::complex<float> alm4 = 0;
    int m5, tmp_lm5;
    float delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
	    if(abs(m1+m2+m3)>l123) continue;
            // Create temporary copy of a_l3m3, taking conjugate if necessary
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            // Iterate over m4 (including negative)
            for(int m4=-l4; m4<=l4; m4++){
              m5 = -m1-m2-m3-m4;
              if (m5<0) continue; // only need to use m5>=0
              if (m5>l5) continue; // this violates triangle conditions
              // Look up the relevant weight
              weight = weight6pcf[n++];
              if (weight==0) continue;
              tmp_lm5 = tmp_l5+m5;
              // Create temporary copies of a_l4m4 and a_l5m5, taking conjugates if necessary
              // No conjugates needed for a_l5m5 since we fixed m5>=0!
              // Note we add the coupling weight factor to a_l5m5
              if (m4 < 0) alm4 = alm3*almconj[almidx+l*nlm+tmp_l4-m4]; else alm4 = alm3*alm[almidx+l*nlm+tmp_l4+m4];
              //calculate delta
              delta = weight*(alm4*alm[almidx+m*nlm+tmp_lm5]).imag(); //odd parity
              //add to this element
              pcf_element += delta;
            }
          }
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            if(abs(m1+m2+m3)>l123) continue;
            // Create temporary copy of a_l3m3, taking conjugate if necessary
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            // Iterate over m4 (including negative)
            for(int m4=-l4; m4<=l4; m4++){
              m5 = -m1-m2-m3-m4;
              if (m5<0) continue; // only need to use m5>=0
              if (m5>l5) continue; // this violates triangle conditions
              // Look up the relevant weight
              weight = weight6pcf[n++];
              if (weight==0) continue;
              tmp_lm5 = tmp_l5+m5;
              // Create temporary copies of a_l4m4 and a_l5m5, taking conjugates if necessary
              // No conjugates needed for a_l5m5 since we fixed m5>=0!
              // Note we add the coupling weight factor to a_l5m5
              if (m4 < 0) alm4 = alm3*almconj[almidx+l*nlm+tmp_l4-m4]; else alm4 = alm3*alm[almidx+l*nlm+tmp_l4+m4];
              //calculate delta
              delta = weight*(alm4*alm[almidx+m*nlm+tmp_lm5]).real(); //even parity
              //add to this element
              pcf_element += delta;
            }
          }
        }
      }
    }
    sixpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power6_kernel_mixed(double *sixpcf, double *weight6pcf,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
	int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
	int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j, int *lut6_k,
	int *lut6_l, int *lut6_m, float wp, int nlm, int nouter,
	int ninner, int almidx) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner) return;
    //compute indices for LUTs
    int iouter = i/ninner;
    int iinner = i%ninner;
    //calc bin_index
    int bin_index = lut6_zeta[iouter]+iinner;
    double pcf_element = sixpcf[bin_index]; // this element
    //cald weight
    double weight = weight6pcf[lut6_n[iouter]];
    //outer loop indices
    int l1 = lut6_l1[iouter];
    int tmp_l1 = l1*(l1+1)/2;
    int l2 = lut6_l2[iouter];
    int tmp_l2 = l2*(l2+1)/2;
    int l12 = lut6_l12[iouter];
    int l3 = lut6_l3[iouter];
    int l123 = lut6_l123[iouter];
    int tmp_l3 = l3*(l3+1)/2;
    int l4 = lut6_l4[iouter];
    int tmp_l4 = l4*(l4+1)/2;
    int l5 = lut6_l5[iouter];
    int tmp_l5 = l5*(l5+1)/2;
    bool odd = lut6_odd[iouter];
    int n = lut6_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut6_i[iinner];
    int j = lut6_j[iinner];
    int k = lut6_k[iinner];
    int l = lut6_l[iinner];
    int m = lut6_m[iinner];
    //alms
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    thrust::complex<float> alm3 = 0;
    thrust::complex<float> alm4 = 0;
    int m5, tmp_lm5;
    double delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    //Put loops inside if (odd) block rather than other way around
    if (odd) {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
	    if(abs(m1+m2+m3)>l123) continue;
            // Create temporary copy of a_l3m3, taking conjugate if necessary
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            // Iterate over m4 (including negative)
            for(int m4=-l4; m4<=l4; m4++){
              m5 = -m1-m2-m3-m4;
              if (m5<0) continue; // only need to use m5>=0
              if (m5>l5) continue; // this violates triangle conditions
              // Look up the relevant weight
              weight = weight6pcf[n++];
              if (weight==0) continue;
              tmp_lm5 = tmp_l5+m5;
              // Create temporary copies of a_l4m4 and a_l5m5, taking conjugates if necessary
              // No conjugates needed for a_l5m5 since we fixed m5>=0!
              // Note we add the coupling weight factor to a_l5m5
              if (m4 < 0) alm4 = alm3*almconj[almidx+l*nlm+tmp_l4-m4]; else alm4 = alm3*alm[almidx+l*nlm+tmp_l4+m4];
              //calculate delta
              delta = weight*(alm4*alm[almidx+m*nlm+tmp_lm5]).imag(); //odd parity
              //add to this element
              pcf_element += delta;
            }
          }
        }
      }
    } else {
      for(int m1=-l1; m1<=l1; m1++){
        // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
        if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
        // Iterate over all m2 (including negative)
        for(int m2=-l2; m2<=l2; m2++){
          if(abs(m1+m2)>l12) continue; // m12 condition
          // Create temporary copy of a_l2m2, taking conjugate if necessary
          if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
          // Iterate over m3 (including negative)
          for(int m3=-l3; m3<=l3; m3++){
            if(abs(m1+m2+m3)>l123) continue;
            // Create temporary copy of a_l3m3, taking conjugate if necessary
            if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
            // Iterate over m4 (including negative)
            for(int m4=-l4; m4<=l4; m4++){
              m5 = -m1-m2-m3-m4;
              if (m5<0) continue; // only need to use m5>=0
              if (m5>l5) continue; // this violates triangle conditions
              // Look up the relevant weight
              weight = weight6pcf[n++];
              if (weight==0) continue;
              tmp_lm5 = tmp_l5+m5;
              // Create temporary copies of a_l4m4 and a_l5m5, taking conjugates if necessary
              // No conjugates needed for a_l5m5 since we fixed m5>=0!
              // Note we add the coupling weight factor to a_l5m5
              if (m4 < 0) alm4 = alm3*almconj[almidx+l*nlm+tmp_l4-m4]; else alm4 = alm3*alm[almidx+l*nlm+tmp_l4+m4];
              //calculate delta
              delta = weight*(alm4*alm[almidx+m*nlm+tmp_lm5]).real(); //even parity
              //add to this element
              pcf_element += delta;
            }
          }
        }
      }
    }
    sixpcf[bin_index] = pcf_element; //copy back to global memory 
}


// ======================================================= /
//  ALL ALM FUNCTIONS ARE HERE                             /
// ======================================================= /

__device__ double CM(double *m, int *map, int mapdim, int md2, int startidx, 
	int a, int b, int c) {
    int idx = a*md2+b*mapdim+c;
    //return m[map[a][b][c]];
    //return m[nmult*i+map[idx]];
    return m[startidx+map[idx]];
}

__global__ void compute_alms(thrust::complex<double>* alm, thrust::complex<double> *almconj, int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nbin*maxp) return;

    //we have i threads = nbin * maxp
    int n = i*nlm; //start index for this thread
    int nst = i*nlm;

    int startidx = nmult*i;
    int md2 = mapdim*mapdim;

    // 0,0:   1
    alm[n++] = CM(m, map, mapdim, md2, startidx, 0,0,0);

    if (order > 0) {
      // 1,0:   z
      // 1,1:   1
      alm[n++] = CM(m, map, mapdim, md2, startidx, 0,0,1);
      alm[n++] = thrust::complex<double>(CM(m, map, mapdim, md2, startidx, 1,0,0),
                    CM(m, map, mapdim, md2, startidx, 0,1,0));
    }

    if (order > 1) {
      // 2,0:   3 z^2 - 1
      // 2,1:   z
      // 2,2:   1
      alm[n++] = 3*CM(m, map, mapdim, md2, startidx, 0,0,2)-CM(m, map, mapdim, md2, startidx, 0,0,0);
      alm[n++] = thrust::complex<double>( CM(m, map, mapdim, md2, startidx, 1,0,1)
                    ,CM(m, map, mapdim, md2, startidx, 0,1,1));
      alm[n++] = thrust::complex<double>( CM(m, map, mapdim, md2, startidx, 2,0,0)
                    -CM(m, map, mapdim, md2, startidx, 0,2,0)
                    ,2*CM(m, map, mapdim, md2, startidx, 1,1,0));
    }

    if (order > 2) {
      // 3,0:   5 z^3 - 3 z
      // 3,1:   5 z^2 - 1
      // 3,2:   z
      // 3,3:   1
      alm[n++] = 5*CM(m, map, mapdim, md2, startidx, 0,0,3)-3*CM(m, map, mapdim, md2, startidx, 0,0,1);
      alm[n++] = thrust::complex<double>(
         5*CM(m, map, mapdim, md2, startidx, 1,0,2)-CM(m, map, mapdim, md2, startidx, 1,0,0)
        ,5*CM(m, map, mapdim, md2, startidx, 0,1,2)-CM(m, map, mapdim, md2, startidx, 0,1,0)
      );
      alm[n++] = thrust::complex<double>(
           CM(m, map, mapdim, md2, startidx, 2,0,1)
          -CM(m, map, mapdim, md2, startidx, 0,2,1)
        ,2*CM(m, map, mapdim, md2, startidx, 1,1,1)
      );
      alm[n++] = thrust::complex<double>(
           CM(m, map, mapdim, md2, startidx, 3,0,0)
        -3*CM(m, map, mapdim, md2, startidx, 1,2,0)
        ,3*CM(m, map, mapdim, md2, startidx, 2,1,0)
          -CM(m, map, mapdim, md2, startidx, 0,3,0)
      );
    }

    if (order > 3) {
      // 4,0:   35 z^4 - 30 z^2 + 3
      // 4,1:   7 z^3 - 3 z
      // 4,2:   7 z^2 - 1
      // 4,3:   z
      // 4,4:   1
      alm[n++] = 35*CM(m, map, mapdim, md2, startidx, 0,0,4)-30*CM(m, map, mapdim, md2, startidx, 0,0,2)+3*CM(m, map, mapdim, md2, startidx, 0,0,0);
      alm[n++] = thrust::complex<double>(
         7*CM(m, map, mapdim, md2, startidx, 1,0,3) - 3*CM(m, map, mapdim, md2, startidx, 1,0,1)
        ,7*CM(m, map, mapdim, md2, startidx, 0,1,3) - 3*CM(m, map, mapdim, md2, startidx, 0,1,1)
      );
      alm[n++] = thrust::complex<double>(
           (7*CM(m, map, mapdim, md2, startidx, 2,0,2)-CM(m, map, mapdim, md2, startidx, 2,0,0))
          -(7*CM(m, map, mapdim, md2, startidx, 0,2,2)-CM(m, map, mapdim, md2, startidx, 0,2,0))
        ,2*(7*CM(m, map, mapdim, md2, startidx, 1,1,2)-CM(m, map, mapdim, md2, startidx, 1,1,0))
       );
      alm[n++] = thrust::complex<double>(
           CM(m, map, mapdim, md2, startidx, 3,0,1)
        -3*CM(m, map, mapdim, md2, startidx, 1,2,1)
        ,3*CM(m, map, mapdim, md2, startidx, 2,1,1)
          -CM(m, map, mapdim, md2, startidx, 0,3,1)
       );
      alm[n++] = thrust::complex<double>(
           CM(m, map, mapdim, md2, startidx, 4,0,0)
        -6*CM(m, map, mapdim, md2, startidx, 2,2,0)
          +CM(m, map, mapdim, md2, startidx, 0,4,0)
        ,4*CM(m, map, mapdim, md2, startidx, 3,1,0)
        -4*CM(m, map, mapdim, md2, startidx, 1,3,0)
       );
    }

    if (order > 4) {
      // 5,0:   63 z^5 - 70 z^3 + 15
      // 5,1:   21 z^4 - 14 z^2 + 1
      // 5,2:   3 z^3 - 1 z
      // 5,3:   9 z^2 - 1
      // 5,4:   z
      // 5,5:   1
      alm[n++] = 63*CM(m, map, mapdim, md2, startidx, 0,0,5)-70*CM(m, map, mapdim, md2, startidx, 0,0,3)+15*CM(m, map, mapdim, md2, startidx, 0,0,1);
      alm[n++] = thrust::complex<double>(
         (21*CM(m, map, mapdim, md2, startidx, 1,0,4) - 14*CM(m, map, mapdim, md2, startidx, 1,0,2) + CM(m, map, mapdim, md2, startidx, 1,0,0))
        ,(21*CM(m, map, mapdim, md2, startidx, 0,1,4) - 14*CM(m, map, mapdim, md2, startidx, 0,1,2) + CM(m, map, mapdim, md2, startidx, 0,1,0))
       );
      alm[n++] = thrust::complex<double>(
           (3*CM(m, map, mapdim, md2, startidx, 2,0,3) - CM(m, map, mapdim, md2, startidx, 2,0,1))
          -(3*CM(m, map, mapdim, md2, startidx, 0,2,3) - CM(m, map, mapdim, md2, startidx, 0,2,1))
        ,2*(3*CM(m, map, mapdim, md2, startidx, 1,1,3) - CM(m, map, mapdim, md2, startidx, 1,1,1))
       );
      alm[n++] = thrust::complex<double>(
           (9*CM(m, map, mapdim, md2, startidx, 3,0,2) - CM(m, map, mapdim, md2, startidx, 3,0,0))
        -3*(9*CM(m, map, mapdim, md2, startidx, 1,2,2) - CM(m, map, mapdim, md2, startidx, 1,2,0))
        ,3*(9*CM(m, map, mapdim, md2, startidx, 2,1,2) - CM(m, map, mapdim, md2, startidx, 2,1,0))
          -(9*CM(m, map, mapdim, md2, startidx, 0,3,2) - CM(m, map, mapdim, md2, startidx, 0,3,0))
       );
      alm[n++] = thrust::complex<double>(
           CM(m, map, mapdim, md2, startidx, 4,0,1)
        -6*CM(m, map, mapdim, md2, startidx, 2,2,1)
          +CM(m, map, mapdim, md2, startidx, 0,4,1)
        ,4*CM(m, map, mapdim, md2, startidx, 3,1,1)
        -4*CM(m, map, mapdim, md2, startidx, 1,3,1)
       );
      alm[n++] = thrust::complex<double>(
            CM(m, map, mapdim, md2, startidx, 5,0,0)
        -10*CM(m, map, mapdim, md2, startidx, 3,2,0)
         +5*CM(m, map, mapdim, md2, startidx, 1,4,0)
        , 5*CM(m, map, mapdim, md2, startidx, 4,1,0)
        -10*CM(m, map, mapdim, md2, startidx, 2,3,0)
           +CM(m, map, mapdim, md2, startidx, 0,5,0)
       );
    }
    //calc conjucates
    for (int j = nst; j < n; j++) almconj[j] = conj(alm[j]);
    return;
}

__global__ void compute_alms_float(thrust::complex<float>* alm, thrust::complex<float> *almconj, int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nbin*maxp) return;

    //we have i threads = nbin * maxp
    int n = i*nlm; //start index for this thread
    int nst = i*nlm;

    int startidx = nmult*i;
    int md2 = mapdim*mapdim;

    // 0,0:   1
    alm[n++] = CM(m, map, mapdim, md2, startidx, 0,0,0);

    if (order > 0) {
      // 1,0:   z
      // 1,1:   1
      alm[n++] = CM(m, map, mapdim, md2, startidx, 0,0,1);
      alm[n++] = thrust::complex<float>(CM(m, map, mapdim, md2, startidx, 1,0,0),
                    CM(m, map, mapdim, md2, startidx, 0,1,0));
    }

    if (order > 1) {
      // 2,0:   3 z^2 - 1
      // 2,1:   z
      // 2,2:   1
      alm[n++] = 3*CM(m, map, mapdim, md2, startidx, 0,0,2)-CM(m, map, mapdim, md2, startidx, 0,0,0);
      alm[n++] = thrust::complex<float>( CM(m, map, mapdim, md2, startidx, 1,0,1)
                    ,CM(m, map, mapdim, md2, startidx, 0,1,1));
      alm[n++] = thrust::complex<float>( CM(m, map, mapdim, md2, startidx, 2,0,0)
                    -CM(m, map, mapdim, md2, startidx, 0,2,0)
                    ,2*CM(m, map, mapdim, md2, startidx, 1,1,0));
    }

    if (order > 2) {
      // 3,0:   5 z^3 - 3 z
      // 3,1:   5 z^2 - 1
      // 3,2:   z
      // 3,3:   1
      alm[n++] = 5*CM(m, map, mapdim, md2, startidx, 0,0,3)-3*CM(m, map, mapdim, md2, startidx, 0,0,1);
      alm[n++] = thrust::complex<float>(
         5*CM(m, map, mapdim, md2, startidx, 1,0,2)-CM(m, map, mapdim, md2, startidx, 1,0,0)
        ,5*CM(m, map, mapdim, md2, startidx, 0,1,2)-CM(m, map, mapdim, md2, startidx, 0,1,0)
      );
      alm[n++] = thrust::complex<float>(
           CM(m, map, mapdim, md2, startidx, 2,0,1)
          -CM(m, map, mapdim, md2, startidx, 0,2,1)
        ,2*CM(m, map, mapdim, md2, startidx, 1,1,1)
      );
      alm[n++] = thrust::complex<float>(
           CM(m, map, mapdim, md2, startidx, 3,0,0)
        -3*CM(m, map, mapdim, md2, startidx, 1,2,0)
        ,3*CM(m, map, mapdim, md2, startidx, 2,1,0)
          -CM(m, map, mapdim, md2, startidx, 0,3,0)
      );
    }

    if (order > 3) {
      // 4,0:   35 z^4 - 30 z^2 + 3
      // 4,1:   7 z^3 - 3 z
      // 4,2:   7 z^2 - 1
      // 4,3:   z
      // 4,4:   1
      alm[n++] = 35*CM(m, map, mapdim, md2, startidx, 0,0,4)-30*CM(m, map, mapdim, md2, startidx, 0,0,2)+3*CM(m, map, mapdim, md2, startidx, 0,0,0);
      alm[n++] = thrust::complex<float>(
         7*CM(m, map, mapdim, md2, startidx, 1,0,3) - 3*CM(m, map, mapdim, md2, startidx, 1,0,1)
        ,7*CM(m, map, mapdim, md2, startidx, 0,1,3) - 3*CM(m, map, mapdim, md2, startidx, 0,1,1)
      );
      alm[n++] = thrust::complex<float>(
           (7*CM(m, map, mapdim, md2, startidx, 2,0,2)-CM(m, map, mapdim, md2, startidx, 2,0,0))
          -(7*CM(m, map, mapdim, md2, startidx, 0,2,2)-CM(m, map, mapdim, md2, startidx, 0,2,0))
        ,2*(7*CM(m, map, mapdim, md2, startidx, 1,1,2)-CM(m, map, mapdim, md2, startidx, 1,1,0))
       );
      alm[n++] = thrust::complex<float>(
           CM(m, map, mapdim, md2, startidx, 3,0,1)
        -3*CM(m, map, mapdim, md2, startidx, 1,2,1)
        ,3*CM(m, map, mapdim, md2, startidx, 2,1,1)
          -CM(m, map, mapdim, md2, startidx, 0,3,1)
       );
      alm[n++] = thrust::complex<float>(
           CM(m, map, mapdim, md2, startidx, 4,0,0)
        -6*CM(m, map, mapdim, md2, startidx, 2,2,0)
          +CM(m, map, mapdim, md2, startidx, 0,4,0)
        ,4*CM(m, map, mapdim, md2, startidx, 3,1,0)
        -4*CM(m, map, mapdim, md2, startidx, 1,3,0)
       );
    }

    if (order > 4) {
      // 5,0:   63 z^5 - 70 z^3 + 15
      // 5,1:   21 z^4 - 14 z^2 + 1
      // 5,2:   3 z^3 - 1 z
      // 5,3:   9 z^2 - 1
      // 5,4:   z
      // 5,5:   1
      alm[n++] = 63*CM(m, map, mapdim, md2, startidx, 0,0,5)-70*CM(m, map, mapdim, md2, startidx, 0,0,3)+15*CM(m, map, mapdim, md2, startidx, 0,0,1);
      alm[n++] = thrust::complex<float>(
         (21*CM(m, map, mapdim, md2, startidx, 1,0,4) - 14*CM(m, map, mapdim, md2, startidx, 1,0,2) + CM(m, map, mapdim, md2, startidx, 1,0,0))
        ,(21*CM(m, map, mapdim, md2, startidx, 0,1,4) - 14*CM(m, map, mapdim, md2, startidx, 0,1,2) + CM(m, map, mapdim, md2, startidx, 0,1,0))
       );
      alm[n++] = thrust::complex<float>(
           (3*CM(m, map, mapdim, md2, startidx, 2,0,3) - CM(m, map, mapdim, md2, startidx, 2,0,1))
          -(3*CM(m, map, mapdim, md2, startidx, 0,2,3) - CM(m, map, mapdim, md2, startidx, 0,2,1))
        ,2*(3*CM(m, map, mapdim, md2, startidx, 1,1,3) - CM(m, map, mapdim, md2, startidx, 1,1,1))
       );
      alm[n++] = thrust::complex<float>(
           (9*CM(m, map, mapdim, md2, startidx, 3,0,2) - CM(m, map, mapdim, md2, startidx, 3,0,0))
        -3*(9*CM(m, map, mapdim, md2, startidx, 1,2,2) - CM(m, map, mapdim, md2, startidx, 1,2,0))
        ,3*(9*CM(m, map, mapdim, md2, startidx, 2,1,2) - CM(m, map, mapdim, md2, startidx, 2,1,0))
          -(9*CM(m, map, mapdim, md2, startidx, 0,3,2) - CM(m, map, mapdim, md2, startidx, 0,3,0))
       );
      alm[n++] = thrust::complex<float>(
           CM(m, map, mapdim, md2, startidx, 4,0,1)
        -6*CM(m, map, mapdim, md2, startidx, 2,2,1)
          +CM(m, map, mapdim, md2, startidx, 0,4,1)
        ,4*CM(m, map, mapdim, md2, startidx, 3,1,1)
        -4*CM(m, map, mapdim, md2, startidx, 1,3,1)
       );
      alm[n++] = thrust::complex<float>(
            CM(m, map, mapdim, md2, startidx, 5,0,0)
        -10*CM(m, map, mapdim, md2, startidx, 3,2,0)
         +5*CM(m, map, mapdim, md2, startidx, 1,4,0)
        , 5*CM(m, map, mapdim, md2, startidx, 4,1,0)
        -10*CM(m, map, mapdim, md2, startidx, 2,3,0)
           +CM(m, map, mapdim, md2, startidx, 0,5,0)
       );
    }
    //calc conjucates
    for (int j = nst; j < n; j++) almconj[j] = conj(alm[j]);
    return;
}


// ======================================================= /
//  PAIRS AND MULTIPOLES ACCUMULATION                      /
// ======================================================= /

// HELPER METHODS

__device__ int test_cell(int3 cell, int3 nside_cuboid, int ncells) {
    if (nside_cuboid.x <= cell.x ||cell.x < 0 ||nside_cuboid.y <= cell.y || cell.y < 0 || nside_cuboid.z <= cell.z || cell.z < 0) {
      return -1;
    }
    int answer = (cell.x*nside_cuboid.y+cell.y)*nside_cuboid.z+cell.z;
    return answer;
}

__device__ int test_cell_periodic(int3 cell, int3 nside_cuboid, int ncells) {
    // Return the 1-d cell number, after wrapping
    // We apply a very large bias, so that we're
    // guaranteed to wrap any reasonable input.
    int cx = (cell.x+ncells)%nside_cuboid.x;
    int cy = (cell.y+ncells)%nside_cuboid.y;
    int cz = (cell.z+ncells)%nside_cuboid.z;
    int answer = (cx*nside_cuboid.y+cy)*nside_cuboid.z+cz;
    return answer;
}

__device__ int3 cell_id_from_1d(int n, int3 nside_cuboid) {
    // Undo 1d back to 3-d indexing
    int3 cid;
    cid.z = n%nside_cuboid.z;
    n = n/nside_cuboid.z;
    cid.y = n%nside_cuboid.y;
    cid.x = n/nside_cuboid.y;
    return cid;
}

/****   Add particles methods ****/
// We have 16 methods for each of pairs_only and pairs_and_multipoles
// DEFAULTS in CAPS
// NORMAL mode or periodic
// original kernel or FAST
// SHARED vs global memory
// DOUBLE vs float


//PAIRS ONLY

__global__ void add_pairs_only_kernel(double *posx, double *posy, double *posz,
	double *w, int *pnum, int *spnum, int *snp, int *sc,
	double *x0i, double *x2i, int n, int nbin, 
        float rmin, float rmax, float rmin2, float rmax2) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    double wj = w[j];
    double pxj = posx[j];
    double pyj = posy[j];
    double pzj = posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
}

__global__ void add_pairs_only_kernel_float(double *posx, double *posy,
	double *posz, double *w, int *pnum, int *spnum, int *snp, int *sc,
	double *x0i, double *x2i, int n, int nbin, 
        float rmin, float rmax, float rmin2, float rmax2) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    float wj = (float)w[j];
    float pxj = (float)posx[j];
    float pyj = (float)posy[j];
    float pzj = (float)posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = (float)posx[k] - pxj;
      dy = (float)posy[k] - pyj;
      dz = (float)posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
}

__global__ void add_pairs_only_kernel_shared(double *posx, double *posy,
	double *posz, double *w, int *pnum, int *spnum, int *snp, int *sc,
	double *x0i, double *x2i, int n, int nbin, 
        float rmin, float rmax, float rmin2, float rmax2) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    double wj = w[j];
    double pxj = posx[j];
    double pyj = posy[j];
    double pzj = posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_only_kernel_float_shared(double *posx, double *posy,
	double *posz, double *w, int *pnum, int *spnum, int *snp, int *sc,
	double *x0i, double *x2i, int n, int nbin, 
        float rmin, float rmax, float rmin2, float rmax2) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    float wj = (float)w[j];
    float pxj = (float)posx[j];
    float pyj = (float)posy[j];
    float pzj = (float)posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = (float)posx[k] - pxj;
      dy = (float)posy[k] - pyj;
      dz = (float)posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_only_kernel_fast(double *posx, double *posy,
	double *posz, double *w, int *start_list, int *np_list,
	int *cellnums, double *x0i, double *x2i, int n, int nbin, int order,
	int nmult, float rmin, float rmax, float rmin2, float rmax2,
	int3 nside_cuboid, int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    double wj = w[j];
    double pxj = posx[j];
    double pyj = posy[j];
    double pzj = posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }

}

__global__ void add_pairs_only_kernel_fast_shared(double *posx, double *posy,
	double *posz, double *w, int *start_list, int *np_list, int *cellnums,
	double *x0i, double *x2i, int n, int nbin, int order, int nmult,
	float rmin, float rmax, float rmin2, float rmax2, int3 nside_cuboid,
	int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin = 0;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);

    //dont return because need to syncthreads
    if (tmp_test >= 0 && tmp_test < ncells) {
      int st = start_list[tmp_test];
      int np = np_list[tmp_test];

      double wj = w[j];
      double pxj = posx[j];
      double pyj = posy[j];
      double pzj = posz[j];

      for (int k = st; k < st+np; k++) {
        if (samecell && j == k) continue;

        dx = posx[k] - pxj;
        dy = posy[k] - pyj;
        dz = posz[k] - pzj;
        norm2 = (dx*dx + dy*dy + dz*dz);
        if (norm2 >= rmax2 || norm2 <= rmin2) continue;

        norm2 = sqrt(norm2);
        bin = floor((norm2-rmin)*bin_factor);
        dz /= norm2;

        // Accumulate the 2-pt correlation function
        //pair_w = w[k]*w[j];
        pair_w = w[k]*wj;
        atomicAdd(&sx0i[bin], pair_w);
        atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_only_kernel_fast_float(double *posx, double *posy,
	double *posz, double *w, int *start_list, int *np_list, int *cellnums,
	double *x0i, double *x2i, int n, int nbin, int order, int nmult,
	float rmin, float rmax, float rmin2, float rmax2, int3 nside_cuboid,
	int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    float wj = (float)w[j];
    float pxj = (float)posx[j];
    float pyj = (float)posy[j];
    float pzj = (float)posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
}

__global__ void add_pairs_only_kernel_fast_float_shared(double *posx,
	double *posy, double *posz, double *w, int *start_list, int *np_list,
	int *cellnums, double *x0i, double *x2i, int n, int nbin, int order,
	int nmult, float rmin, float rmax, float rmin2, float rmax2,
	int3 nside_cuboid, int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin = 0;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);

    //dont return because need to syncthreads
    if (tmp_test >= 0 && tmp_test < ncells) {
      int st = start_list[tmp_test];
      int np = np_list[tmp_test];

      float wj = (float)w[j];
      float pxj = (float)posx[j];
      float pyj = (float)posy[j];
      float pzj = (float)posz[j];

      for (int k = st; k < st+np; k++) {
        if (samecell && j == k) continue;

        dx = posx[k] - pxj;
        dy = posy[k] - pyj;
        dz = posz[k] - pzj;
        norm2 = (dx*dx + dy*dy + dz*dz);
        if (norm2 >= rmax2 || norm2 <= rmin2) continue;

        norm2 = sqrt(norm2);
        bin = floor((norm2-rmin)*bin_factor);
        dz /= norm2;

        // Accumulate the 2-pt correlation function
        //pair_w = w[k]*w[j];
        pair_w = w[k]*wj;
        atomicAdd(&sx0i[bin], pair_w);
        atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_only_periodic_kernel(double *posx, double *posy,
	double *posz, double *w, int *pnum, int *spnum, int *snp, int *sc,
	double *x0i, double *x2i, int *delta_x, int *delta_y, int *delta_z,
	int n, int nbin, float rmin, float rmax, float rmin2, float rmax2,
	double cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-delta_x[i]*cellsize;
    ppos_y = posy[j]-delta_y[i]*cellsize;
    ppos_z = posz[j]-delta_z[i]*cellsize;
    double wj = w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x;
      dy = posy[k] - ppos_y;
      dz = posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
}

__global__ void add_pairs_only_periodic_kernel_float(double *posx,
	double *posy, double *posz, double *w, int *pnum, int *spnum,
	int *snp, int *sc, double *x0i, double *x2i,
        int *delta_x, int *delta_y, int *delta_z, int n, int nbin, 
        float rmin, float rmax, float rmin2, float rmax2, float cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)posx[j]-delta_x[i]*cellsize;
    ppos_y = (float)posy[j]-delta_y[i]*cellsize;
    ppos_z = (float)posz[j]-delta_z[i]*cellsize;
    float wj = (float)w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x;
      dy = (float)posy[k] - ppos_y;
      dz = (float)posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
}

__global__ void add_pairs_only_periodic_kernel_shared(double *posx,
	double *posy, double *posz, double *w, int *pnum, int *spnum,
	int *snp, int *sc, double *x0i, double *x2i,
        int *delta_x, int *delta_y, int *delta_z, int n, int nbin, 
        float rmin, float rmax, float rmin2, float rmax2, double cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-delta_x[i]*cellsize;
    ppos_y = posy[j]-delta_y[i]*cellsize;
    ppos_z = posz[j]-delta_z[i]*cellsize;
    double wj = w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x;
      dy = posy[k] - ppos_y;
      dz = posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_only_periodic_kernel_float_shared(double *posx,
	double *posy, double *posz, double *w, int *pnum, int *spnum,
	int *snp, int *sc, double *x0i, double *x2i,
        int *delta_x, int *delta_y, int *delta_z, int n, int nbin, 
        float rmin, float rmax, float rmin2, float rmax2, float cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)posx[j]-delta_x[i]*cellsize;
    ppos_y = (float)posy[j]-delta_y[i]*cellsize;
    ppos_z = (float)posz[j]-delta_z[i]*cellsize;
    float wj = (float)w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x;
      dy = (float)posy[k] - ppos_y;
      dz = (float)posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_only_periodic_kernel_fast(double *posx, double *posy,
	double *posz, double *w, int *start_list, int *np_list, int *cellnums,
	double *x0i, double *x2i, int n, int nbin, int order, int nmult,
	float rmin, float rmax, float rmin2, float rmax2, int3 nside_cuboid,
	int ncells, int maxsep, int pstart, double cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-deltax*cellsize;
    ppos_y = posy[j]-deltay*cellsize;
    ppos_z = posz[j]-deltaz*cellsize;
    double wj = w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x;
      dy = posy[k] - ppos_y;
      dz = posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
}

__global__ void add_pairs_only_periodic_kernel_fast_shared(double *posx,
	double *posy, double *posz, double *w, int *start_list, int *np_list,
	int *cellnums, double *x0i, double *x2i, int n, int nbin, int order,
	int nmult, float rmin, float rmax, float rmin2, float rmax2,
	int3 nside_cuboid, int ncells, int maxsep, int pstart,
	double cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-deltax*cellsize;
    ppos_y = posy[j]-deltay*cellsize;
    ppos_z = posz[j]-deltaz*cellsize;
    double wj = w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x;
      dy = posy[k] - ppos_y;
      dz = posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_only_periodic_kernel_fast_float(double *posx,
	double *posy, double *posz, double *w, int *start_list, int *np_list,
	int *cellnums, double *x0i, double *x2i, int n, int nbin, int order,
	int nmult, float rmin, float rmax, float rmin2, float rmax2,
	int3 nside_cuboid, int ncells, int maxsep, int pstart,
	float cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)posx[j]-deltax*cellsize;
    ppos_y = (float)posy[j]-deltay*cellsize;
    ppos_z = (float)posz[j]-deltaz*cellsize;
    float wj = (float)w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x;
      dy = (float)posy[k] - ppos_y;
      dz = (float)posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
}

__global__ void add_pairs_only_periodic_kernel_fast_float_shared(double *posx,
	double *posy, double *posz, double *w, int *start_list, int *np_list,
	int *cellnums, double *x0i, double *x2i, int n, int nbin, int order,
	int nmult, float rmin, float rmax, float rmin2, float rmax2,
	int3 nside_cuboid, int ncells, int maxsep, int pstart,
	float cellsize) {
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)posx[j]-deltax*cellsize;
    ppos_y = (float)posy[j]-deltay*cellsize;
    ppos_z = (float)posz[j]-deltaz*cellsize;
    float wj = (float)w[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x;
      dy = (float)posy[k] - ppos_y;
      dz = (float)posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

// ======================================================= /
//PAIRS AND MULTIPOLES

__global__ void add_pairs_and_multipoles_kernel(double *m, double *posx,
	double *posy, double *posz, double *w, int *ct, int *pnum, int *spnum,
	int *snp, int *sc, double *x0i, double *x2i, int n, int nbin,
	int order, int nmult, float rmin, float rmax, float rmin2,
	float rmax2, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    double wj = w[j];
    double pxj = posx[j];
    double pyj = posy[j];
    double pzj = posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      //pair_w = w[k]*w[j];
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      double fi, fij, fijk;
      int midx = 0;

      double sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
	    atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
}

__global__ void add_pairs_and_multipoles_kernel_shared(double *m, double *posx,
	double *posy, double *posz, double *w, int *ct, int *pnum, int *spnum,
	int *snp, int *sc, double *x0i, double *x2i, int n, int nbin,
	int order, int nmult, float rmin, float rmax, float rmin2,
	float rmax2, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    double wj = w[j];
    double pxj = posx[j];
    double pyj = posy[j];
    double pzj = posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      //pair_w = w[k]*w[j];
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      double fi, fij, fijk;
      int midx = 0;

      double sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
	    atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_and_multipoles_kernel_float(double *m, double *posx,
        double *posy, double *posz, double *w, int *ct, int *pnum, int *spnum,
        int *snp, int *sc, double *x0i, double *x2i, int n, int nbin,
        int order, int nmult, float rmin, float rmax, float rmin2,
        float rmax2, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    float wj = (float)w[j];
    float pxj = (float)posx[j];
    float pyj = (float)posy[j];
    float pzj = (float)posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = (float)posx[k] - pxj;
      dy = (float)posy[k] - pyj;
      dz = (float)posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      //pair_w = w[k]*w[j];
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      float fi, fij, fijk;
      int midx = 0;

      float sum = 0;

      fi = (float)w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
}

__global__ void add_pairs_and_multipoles_kernel_float_shared(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *pnum, int *spnum, int *snp, int *sc, double *x0i, double *x2i,
	int n, int nbin, int order, int nmult, float rmin, float rmax,
	float rmin2, float rmax2, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    float wj = (float)w[j];
    float pxj = (float)posx[j];
    float pyj = (float)posy[j];
    float pzj = (float)posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;
      dx = (float)posx[k] - pxj;
      dy = (float)posy[k] - pyj;
      dz = (float)posz[k] - pzj;

      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      //pair_w = w[k]*w[j];
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      float fi, fij, fijk;
      int midx = 0;

      float sum = 0;

      fi = (float)w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}


__global__ void add_pairs_and_multipoles_kernel_fast(double *m, double *posx,
	double *posy, double *posz, double *w, int *ct, int *start_list,
	int *np_list, int *cellnums, double *x0i, double *x2i, int n,
	int nbin, int order, int nmult, float rmin, float rmax, float rmin2,
	float rmax2, int3 nside_cuboid, int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    double wj = w[j];
    double pxj = posx[j];
    double pyj = posy[j];
    double pzj = posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      double fi, fij, fijk;
      int midx = 0;

      double sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
      fij = fi;
      for (int jj = 0; jj <= order-ii; jj++) {
        fijk = fij;
        for (int kk = 0; kk <= order-ii-jj; kk++) {
          sum += fijk;
          fijk *= dz;
          //now incrementing to next index - copy sum to this index
          atomicAdd(&m[idx2+midx], sum);
          sum = 0;
          midx++;
        }
        fij *= dy;
      }
      fi *= dx;
      }
    }

}

__global__ void add_pairs_and_multipoles_kernel_fast_shared(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, float rmin2, float rmax2, int3 nside_cuboid,
	int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin = 0;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);

    //dont return because need to syncthreads
    if (tmp_test >= 0 && tmp_test < ncells) {
      int st = start_list[tmp_test];
      int np = np_list[tmp_test];

      //take multiplication out of loop
      int idx1 = (j-pstart)*nbin*nmult;
      int idx2;
      int cidx = (j-pstart)*nbin;

      double wj = w[j];
      double pxj = posx[j];
      double pyj = posy[j];
      double pzj = posz[j];

      for (int k = st; k < st+np; k++) {
        if (samecell && j == k) continue;

        dx = posx[k] - pxj;
        dy = posy[k] - pyj;
        dz = posz[k] - pzj;
        norm2 = (dx*dx + dy*dy + dz*dz);
        if (norm2 >= rmax2 || norm2 <= rmin2) continue;

        norm2 = sqrt(norm2);
        bin = floor((norm2-rmin)*bin_factor);

        //take multiplication out of loop
        idx2 = idx1+bin*nmult;
        dx /= norm2;
        dy /= norm2;
        dz /= norm2;

        // Accumulate the 2-pt correlation function
        //pair_w = w[k]*w[j];
        pair_w = w[k]*wj;
        atomicAdd(&sx0i[bin], pair_w);
        atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

        //Multipoles
        atomicAdd(&ct[cidx+bin], 1);

        double fi, fij, fijk;
        int midx = 0;

        double sum = 0;

        fi = w[k];
        for (int ii = 0; ii <= order; ii++) {
          fij = fi;
          for (int jj = 0; jj <= order-ii; jj++) {
            fijk = fij;
            for (int kk = 0; kk <= order-ii-jj; kk++) {
              sum += fijk;
              fijk *= dz;
              //now incrementing to next index - copy sum to this index
              //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
              atomicAdd(&m[idx2+midx], sum);
              sum = 0;
              midx++;
            }
            fij *= dy;
          }
          fi *= dx;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_and_multipoles_kernel_fast_float(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, float rmin2, float rmax2, int3 nside_cuboid,
	int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    float wj = (float)w[j];
    float pxj = (float)posx[j];
    float pyj = (float)posy[j];
    float pzj = (float)posz[j];

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - pxj;
      dy = posy[k] - pyj;
      dz = posz[k] - pzj;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      float fi, fij, fijk;
      int midx = 0;

      float sum = 0;

      fi = (float)w[k];
      for (int ii = 0; ii <= order; ii++) {
      fij = fi;
      for (int jj = 0; jj <= order-ii; jj++) {
        fijk = fij;
        for (int kk = 0; kk <= order-ii-jj; kk++) {
          sum += fijk;
          fijk *= dz;
          //now incrementing to next index - copy sum to this index
          atomicAdd(&m[idx2+midx], sum);
          sum = 0;
          midx++;
        }
        fij *= dy;
      }
      fi *= dx;
      }
    }
}

__global__ void add_pairs_and_multipoles_kernel_fast_float_shared(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, float rmin2, float rmax2, int3 nside_cuboid,
	int ncells, int maxsep, int pstart) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin = 0;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell(cell_loc, nside_cuboid, ncells);

    //dont return because need to syncthreads
    if (tmp_test >= 0 && tmp_test < ncells) {
      int st = start_list[tmp_test];
      int np = np_list[tmp_test];

      //take multiplication out of loop
      int idx1 = (j-pstart)*nbin*nmult;
      int idx2;
      int cidx = (j-pstart)*nbin;

      float wj = (float)w[j];
      float pxj = (float)posx[j];
      float pyj = (float)posy[j];
      float pzj = (float)posz[j];

      for (int k = st; k < st+np; k++) {
        if (samecell && j == k) continue;

        dx = posx[k] - pxj;
        dy = posy[k] - pyj;
        dz = posz[k] - pzj;
        norm2 = (dx*dx + dy*dy + dz*dz);
        if (norm2 >= rmax2 || norm2 <= rmin2) continue;

        norm2 = sqrt(norm2);
        bin = floor((norm2-rmin)*bin_factor);

        //take multiplication out of loop
        idx2 = idx1+bin*nmult;
        dx /= norm2;
        dy /= norm2;
        dz /= norm2;

        // Accumulate the 2-pt correlation function
        //pair_w = w[k]*w[j];
        pair_w = w[k]*wj;
        atomicAdd(&sx0i[bin], pair_w);
        atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

        //Multipoles
        atomicAdd(&ct[cidx+bin], 1);

        float fi, fij, fijk;
        int midx = 0;

        float sum = 0;

        fi = (float)w[k];
        for (int ii = 0; ii <= order; ii++) {
          fij = fi;
          for (int jj = 0; jj <= order-ii; jj++) {
            fijk = fij;
            for (int kk = 0; kk <= order-ii-jj; kk++) {
              sum += fijk;
              fijk *= dz;
              //now incrementing to next index - copy sum to this index
              //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
              atomicAdd(&m[idx2+midx], sum);
              sum = 0;
              midx++;
            }
            fij *= dy;
          }
          fi *= dx;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_and_multipoles_periodic_kernel(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *pnum, int *spnum, int *snp, int *sc, double *x0i, double *x2i,
	int *delta_x, int *delta_y, int *delta_z, int n, int nbin, int order,
	int nmult, float rmin, float rmax, float rmin2, float rmax2,
	int pstart, double cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    
    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-delta_x[i]*cellsize;
    ppos_y = posy[j]-delta_y[i]*cellsize;
    ppos_z = posz[j]-delta_z[i]*cellsize;
    double wj = w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x; 
      dy = posy[k] - ppos_y; 
      dz = posz[k] - ppos_z; 
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      double fi, fij, fijk;
      int midx = 0;

      double sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
}

__global__ void add_pairs_and_multipoles_periodic_kernel_shared(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *pnum, int *spnum, int *snp, int *sc, double *x0i, double *x2i,
	int *delta_x, int *delta_y, int *delta_z, int n, int nbin, int order,
	int nmult, float rmin, float rmax, float rmin2, float rmax2,
	int pstart, double cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-delta_x[i]*cellsize;
    ppos_y = posy[j]-delta_y[i]*cellsize;
    ppos_z = posz[j]-delta_z[i]*cellsize;
    double wj = w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x; 
      dy = posy[k] - ppos_y; 
      dz = posz[k] - ppos_z; 
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      double fi, fij, fijk;
      int midx = 0;

      double sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_and_multipoles_periodic_kernel_float(double *m,
        double *posx, double *posy, double *posz, double *w, int *ct,
        int *pnum, int *spnum, int *snp, int *sc, double *x0i, double *x2i,
        int *delta_x, int *delta_y, int *delta_z, int n, int nbin, int order,
        int nmult, float rmin, float rmax, float rmin2, float rmax2,
        int pstart, float cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)(posx[j]-delta_x[i]*cellsize);
    ppos_y = (float)(posy[j]-delta_y[i]*cellsize);
    ppos_z = (float)(posz[j]-delta_z[i]*cellsize);
    float wj = (float)w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x;
      dy = (float)posy[k] - ppos_y;
      dz = (float)posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      float fi, fij, fijk;
      int midx = 0;

      float sum = 0;

      fi = (float)w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
}

__global__ void add_pairs_and_multipoles_periodic_kernel_float_shared(double *m,
        double *posx, double *posy, double *posz, double *w, int *ct,
        int *pnum, int *spnum, int *snp, int *sc, double *x0i, double *x2i,
        int *delta_x, int *delta_y, int *delta_z, int n, int nbin, int order,
        int nmult, float rmin, float rmax, float rmin2, float rmax2,
        int pstart, float cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //pnum = primary particle idx (j) = length i
    //spnum = secondary particle idx (k) = length i
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();

    int samecell = sc[i];
    int j = pnum[i];
    int st = spnum[i];
    int np = snp[i];

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)(posx[j]-delta_x[i]*cellsize);
    ppos_y = (float)(posy[j]-delta_y[i]*cellsize);
    ppos_z = (float)(posz[j]-delta_z[i]*cellsize);
    float wj = (float)w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x;
      dy = (float)posy[k] - ppos_y;
      dz = (float)posz[k] - ppos_z;
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      float fi, fij, fijk;
      int midx = 0;

      float sum = 0;

      fi = (float)w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_and_multipoles_periodic_kernel_fast(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, float rmin2, float rmax2, int3 nside_cuboid, int ncells,
	int maxsep, int pstart, double cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    
    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-deltax*cellsize;
    ppos_y = posy[j]-deltay*cellsize;
    ppos_z = posz[j]-deltaz*cellsize;
    double wj = w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x; 
      dy = posy[k] - ppos_y; 
      dz = posz[k] - ppos_z; 
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      double fi, fij, fijk;
      int midx = 0;

      double sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
}

__global__ void add_pairs_and_multipoles_periodic_kernel_fast_shared(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, float rmin2, float rmax2, int3 nside_cuboid, int ncells,
	int maxsep, int pstart, double cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ double sx0i[NBIN];
    __shared__ double sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    double dx, dy, dz, norm2;
    double bin_factor = (double)nbin/(rmax-rmin);
    double pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    double ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = posx[j]-deltax*cellsize;
    ppos_y = posy[j]-deltay*cellsize;
    ppos_z = posz[j]-deltaz*cellsize;
    double wj = w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - ppos_x; 
      dy = posy[k] - ppos_y; 
      dz = posz[k] - ppos_z; 
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      double fi, fij, fijk;
      int midx = 0;

      double sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void add_pairs_and_multipoles_periodic_kernel_fast_float(double *m,
	double *posx, double *posy, double *posz, double *w, int *ct,
	int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, float rmin2, float rmax2, int3 nside_cuboid, int ncells,
	int maxsep, int pstart, double cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    
    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)posx[j]-deltax*cellsize;
    ppos_y = (float)posy[j]-deltay*cellsize;
    ppos_z = (float)posz[j]-deltaz*cellsize;
    float wj = (float)w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x; 
      dy = (float)posy[k] - ppos_y; 
      dz = (float)posz[k] - ppos_z; 
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&x0i[bin], pair_w);
      atomicAdd(&x2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      float fi, fij, fijk;
      int midx = 0;

      float sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
}

__global__ void add_pairs_and_multipoles_periodic_kernel_fast_float_shared(
	double *m, double *posx, double *posy, double *posz, double *w,
	int *ct, int *start_list, int *np_list, int *cellnums, double *x0i,
	double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, float rmin2, float rmax2, int3 nside_cuboid, int ncells,
	int maxsep, int pstart, double cellsize) {
    //m = np * nbin * nmult length
    //ct = np * nbin length
    //posx, posy, posz, ww = length np (every particle)
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    __shared__ float sx0i[NBIN];
    __shared__ float sx2i[NBIN];
    if (threadIdx.x < nbin) {
      sx0i[threadIdx.x] = 0;
      sx2i[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int cellrange = 2*maxsep+1;
    int divisor = cellrange*cellrange*cellrange;
    int j = i/divisor+pstart; //particle index
    int delta = i % divisor; //0-342 if maxsep == 3

    int3 prim_id = cell_id_from_1d(cellnums[j], nside_cuboid);
    int deltaz = delta % cellrange - maxsep;
    int deltay = (delta / cellrange) % cellrange - maxsep;
    int deltax = delta / (cellrange*cellrange) - maxsep;

    int3 cell_loc;

    int bin;
    float dx, dy, dz, norm2;
    float bin_factor = (float)nbin/(rmax-rmin);
    float pair_w;

    int samecell = (deltax == 0 && deltay == 0 && deltaz == 0)?1:0;
    cell_loc.x = prim_id.x + deltax;
    cell_loc.y = prim_id.y + deltay;
    cell_loc.z = prim_id.z + deltaz;

    int tmp_test = test_cell_periodic(cell_loc, nside_cuboid, ncells);
    if(tmp_test < 0 || tmp_test >= ncells) return;

    int st = start_list[tmp_test];
    int np = np_list[tmp_test];

    //periodic calcs
    float ppos_x, ppos_y, ppos_z; //primary particle pos
    ppos_x = (float)posx[j]-deltax*cellsize;
    ppos_y = (float)posy[j]-deltay*cellsize;
    ppos_z = (float)posz[j]-deltaz*cellsize;
    float wj = (float)w[j];

    //take multiplication out of loop
    int idx1 = (j-pstart)*nbin*nmult;
    int idx2;
    int cidx = (j-pstart)*nbin;

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = (float)posx[k] - ppos_x; 
      dy = (float)posy[k] - ppos_y; 
      dz = (float)posz[k] - ppos_z; 
      norm2 = (dx*dx + dy*dy + dz*dz);
      if (norm2 >= rmax2 || norm2 <= rmin2) continue;

      norm2 = sqrt(norm2);
      //bin = floor((norm2-rmin)/(rmax-rmin)*nbin);
      bin = floor((norm2-rmin)*bin_factor);
      //take multiplication out of loop
      idx2 = idx1+bin*nmult;
      dx /= norm2;
      dy /= norm2;
      dz /= norm2;

      // Accumulate the 2-pt correlation function
      pair_w = w[k]*wj;
      atomicAdd(&sx0i[bin], pair_w);
      atomicAdd(&sx2i[bin], pair_w*(3.0*dz*dz-1)*0.5);

      //Multipoles
      atomicAdd(&ct[cidx+bin], 1);

      float fi, fij, fijk;
      int midx = 0;

      float sum = 0;

      fi = w[k];
      for (int ii = 0; ii <= order; ii++) {
        fij = fi;
        for (int jj = 0; jj <= order-ii; jj++) {
          fijk = fij;
          for (int kk = 0; kk <= order-ii-jj; kk++) {
            sum += fijk;
            fijk *= dz;
            //now incrementing to next index - copy sum to this index
            //atomicAdd(&m[j*nbin*nmult+bin*nmult+midx], sum);
            atomicAdd(&m[idx2+midx], sum);
            sum = 0;
            midx++;
          }
          fij *= dy;
        }
        fi *= dx;
      }
    }
    __syncthreads();
    if (threadIdx.x < nbin) {
      atomicAdd(&x0i[threadIdx.x], sx0i[threadIdx.x]);
      atomicAdd(&x2i[threadIdx.x], sx2i[threadIdx.x]);
    }
    __syncthreads();
}


// ======================================================= /
//  CPU METHODS                                            /
// ======================================================= /


// ======================================================= /
//  ALL ADD_TO_POWER FUNCTIONS ARE HERE                    /
// ======================================================= /

//* ==== ADD TO POWER 3 METHODS ===== *//
//3PCF kernels
//Only 1 kernel option for 3CF, 3 precision modes

void gpu_add_to_power3_orig(double *d_threepcf, double *d_weight3pcf,
        double *weights, int *lut3_i, int *lut3_j, int *lut3_ct, 
        int nb, int nlm, int nouter, int order, int np) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  long threads = nouter*np;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power3_kernel_orig<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_threepcf,
        d_weight3pcf, weights, d_alm, d_almconj, lut3_i, lut3_j, lut3_ct,
        nb, nlm, nouter, order, np, pstart3);

  //increment pstart3 in case of data chunking
  pstart3+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power3_orig_float(float *d_threepcf, float *d_weight3pcf,
        double *weights, int *lut3_i, int *lut3_j, int *lut3_ct,
        int nb, int nlm, int nouter, int order, int np) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = nouter*np;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power3_kernel_orig_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
	d_threepcf, d_weight3pcf, weights, f_alm, f_almconj, lut3_i, lut3_j,
	lut3_ct, nb, nlm, nouter, order, np, pstart3);

  //increment pstart3 in case of data chunking
  pstart3+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power3_orig_mixed(double *d_threepcf, double *d_weight3pcf,
        double *weights, int *lut3_i, int *lut3_j, int *lut3_ct,
        int nb, int nlm, int nouter, int order, int np) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = nouter*np;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power3_kernel_orig_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
	d_threepcf, d_weight3pcf, weights, f_alm, f_almconj, lut3_i, lut3_j,
	lut3_ct, nb, nlm, nouter, order, np, pstart3);

  //increment pstart3 in case of data chunking
  pstart3+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//* ==== DISCONNECTED ADD TO POWER 4 METHODS ===== *//
//DISCONNECTED 4PCF kernels
//Only 1 kernel option for 3CF, 3 precision modes

void gpu_add_to_power_discon1_orig(double *d_discon1_r, double *d_discon1_i,
	double *d_weightdiscon, double *weights, int *lut_discon_ell,
	int *lut_discon_mm, int nb, int nlm, int ndiscon1, int order, int np) {
  //d_alm and d_almconj already allocated and computed
  //d_discon1 too

  // Invoke kernel
  long threads = ndiscon1*np;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
std::cout << "THREADS " << threads << std::endl;

  add_to_power_discon1_kernel_orig<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
	d_discon1_r, d_discon1_i, d_weightdiscon, weights, d_alm, d_almconj,
	lut_discon_ell, lut_discon_mm, nb, nlm, ndiscon1, order, np,
	pstart_discon); 

  //increment pstart_discon in case of data chunking
  pstart_discon+=np;

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called from NPCF.h 
}

void gpu_add_to_power_discon1_orig_float(float *f_discon1_r,
        float *f_discon1_i, float *f_weightdiscon,
	double *weights, int *lut_discon_ell, int *lut_discon_mm, int nb,
	int nlm, int ndiscon1, int order, int np) {
  //f_alm and f_almconj already allocated and computed
  //f_discon1 too

  // Invoke kernel
  long threads = ndiscon1*np;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power_discon1_kernel_orig_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        f_discon1_r, f_discon1_i, f_weightdiscon, weights, f_alm, f_almconj,
	lut_discon_ell, lut_discon_mm, nb, nlm, ndiscon1, order, np,
	pstart_discon);

  //increment pstart_discon in case of data chunking
  pstart_discon+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called from NPCF.h 
}

void gpu_add_to_power_discon1_orig_mixed(double *d_discon1_r,
        double *d_discon1_i, double *d_weightdiscon,
	double *weights, int *lut_discon_ell, int *lut_discon_mm, int nb,
	int nlm, int ndiscon1, int order, int np) {
  //f_alm and f_almconj already allocated and computed
  //d_discon1 too

  // Invoke kernel
  long threads = ndiscon1*np;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power_discon1_kernel_orig_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_discon1_r, d_discon1_i, d_weightdiscon, weights, f_alm, f_almconj,
	lut_discon_ell, lut_discon_mm, nb, nlm, ndiscon1, order, np,
	pstart_discon);

  //increment pstart_discon in case of data chunking
  pstart_discon+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called from NPCF.h 
}

//* ==== DISCON2 TERMS ===== *//

void gpu_add_to_power_discon2_orig(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double *weights, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int nb, int nlm, int nouter, int order, int ninner, int np) {
  //d_alm and d_almconj already allocated and computed
  //d_discon1 too

  // Invoke kernel
  long threads = nouter*np; 
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
std::cout << "THREADS2 " << threads << std::endl;

  add_to_power_discon2_kernel_orig<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_discon2_r, d_discon2_i, d_weightdiscon, weights, d_alm, d_almconj,
        lut_discon_ell1, lut_discon_ell2, lut_discon_mm1, lut_discon_mm2,
	nb, nlm, nouter, order, ninner, np, pstart_discon2);

  //increment pstart_discon in case of data chunking
  pstart_discon2+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called from NPCF.h 
}

void gpu_add_to_power_discon2_b(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double wp, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner) {

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

if (pstart_discon2 == 0) {
std::cout << "D2B Threads = " << threads << " Nouter = " << nouter << " Ninner = " << ninner << std::endl;
}

  //calculate index of d_alm for this particle
  int almidx = pstart_discon2*nb*nlm;
  pstart_discon2++;

  add_to_power_discon2_kernel_b<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
	d_discon2_r, d_discon2_i, d_weightdiscon, wp, d_alm, d_almconj,
	lut_discon_ell1, lut_discon_ell2, lut_discon_mm1, lut_discon_mm2,
	lut_discon_i, lut_discon_j,
        nb, nlm, nouter, order, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power_discon2_final(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double *weights, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner, int np,
	int qbalance, int qinvert) {

  // Invoke kernel
  int npblocks = (np/DISCON2_PARTICLES_PER_THREAD)+1;
  int nprnd = npblocks*DISCON2_PARTICLES_PER_THREAD;
  long threads = ninner*nouter*npblocks;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power_discon2_kernel_final<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_discon2_r, d_discon2_i, d_weightdiscon, weights, d_alm, d_almconj,
        lut_discon_ell1, lut_discon_ell2, lut_discon_mm1, lut_discon_mm2,
        lut_discon_i, lut_discon_j,
        nb, nlm, nouter, order, ninner, np, nprnd, npblocks, pstart_discon2,
	qbalance, qinvert);

  //increment pstart_discon in case of data chunking
  pstart_discon2+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power_discon2_final_float(float *f_discon2_r,
	float *f_discon2_i, float *f_weightdiscon, double *weights,
	int *lut_discon_ell1, int *lut_discon_ell2, int *lut_discon_mm1,
	int *lut_discon_mm2, int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner, int np,
	int qbalance, int qinvert) {
  //f_alm and f_almconj already allocated and computed
  // Invoke kernel
  int npblocks = (np/DISCON2_PARTICLES_PER_THREAD)+1;
  int nprnd = npblocks*DISCON2_PARTICLES_PER_THREAD;
  long threads = ninner*nouter*npblocks;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power_discon2_kernel_final_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        f_discon2_r, f_discon2_i, f_weightdiscon, weights, f_alm, f_almconj,
        lut_discon_ell1, lut_discon_ell2, lut_discon_mm1, lut_discon_mm2,
        lut_discon_i, lut_discon_j,
        nb, nlm, nouter, order, ninner, np, nprnd, npblocks, pstart_discon2,
	qbalance, qinvert);

  //increment pstart_discon in case of data chunking
  pstart_discon2+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power_discon2_final_mixed(double *d_discon2_r, double *d_discon2_i,
        double *d_weightdiscon, double *weights, int *lut_discon_ell1,
        int *lut_discon_ell2, int *lut_discon_mm1, int *lut_discon_mm2,
        int *lut_discon_i, int *lut_discon_j,
        int nb, int nlm, int nouter, int order, int ninner, int np,
	int qbalance, int qinvert) {
  //f_alm and f_almconj already allocated and computed
  // Invoke kernel
  int npblocks = (np/DISCON2_PARTICLES_PER_THREAD)+1;
  int nprnd = npblocks*DISCON2_PARTICLES_PER_THREAD;
  long threads = ninner*nouter*npblocks;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  add_to_power_discon2_kernel_final_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_discon2_r, d_discon2_i, d_weightdiscon, weights, f_alm, f_almconj,
        lut_discon_ell1, lut_discon_ell2, lut_discon_mm1, lut_discon_mm2,
        lut_discon_i, lut_discon_j,
        nb, nlm, nouter, order, ninner, np, nprnd, npblocks, pstart_discon2,
	qbalance, qinvert);

  //increment pstart_discon in case of data chunking
  pstart_discon2+=np;

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}


//* ==== ADD TO POWER 4 METHODS ===== *//
//4PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes

void gpu_add_to_power4(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd, int *lut4_n,
	int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k, double wp,
	int nb, int nlm, int nouter, int ninner, int nell4) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

if (count == 0) {
count++;
std::cout << "Threads = " << threads << " Nouter = " << nouter << " Ninner = " << ninner << std::endl;
}

  add_to_power4_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_odd, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k, 
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//float version of main kernel
void gpu_add_to_power4_float(float *d_fourpcf, float *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_odd, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//mixed precision
void gpu_add_to_power4_mixed(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_odd, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//alternate (original) kernel
void gpu_add_to_power4_orig(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd, int *lut4_m1,
	int *lut4_m2, int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j,
	int *lut4_k, double wp, int nb, int nlm, int nouter, int ninner,
	int nell4) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

if (count == 0) {
count++;
std::cout << "Threads = " << threads << std::endl;
}

  add_to_power4_kernel_orig<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2, lut4_l3, lut4_odd,
	lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//float version
void gpu_add_to_power4_orig_float(float *d_fourpcf, float *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_orig_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2, lut4_l3, lut4_odd,
        lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//mixed precision
void gpu_add_to_power4_orig_mixed(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_orig_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2, lut4_l3, lut4_odd,
	lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//* ==== ADD TO POWER 5 METHODS ===== *//
//5PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes

void gpu_add_to_power5(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

if (count == 1) {
count++;
std::cout << "Threads = " << threads << " Nouter = " << nouter << " Ninner = " << ninner << std::endl;
}

  add_to_power5_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2,
        lut5_l12, lut5_l3, lut5_l4, lut5_odd,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_float(float *d_fivepcf, float *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fivepcf,
	d_weight5pcf, f_alm, f_almconj, lut5_l1, lut5_l2, lut5_l12,
	lut5_l3, lut5_l4, lut5_odd, lut5_n, lut5_zeta, lut5_i,
	lut5_j, lut5_k, lut5_l, wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_mixed(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fivepcf,
	d_weight5pcf, f_alm, f_almconj, lut5_l1, lut5_l2, lut5_l12,
	lut5_l3, lut5_l4, lut5_odd, lut5_n, lut5_zeta, lut5_i, lut5_j,
	lut5_k, lut5_l, wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_orig(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

if (count == 1) {
count++;
std::cout << "Threads = " << threads << " Nouter = " << nouter << " Ninner = " << ninner << std::endl;
}

  add_to_power5_kernel_orig<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_odd, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_orig_float(float *d_fivepcf, float *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_orig_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fivepcf,
        d_weight5pcf, f_alm, f_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_odd, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_orig_mixed(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4,
	bool *lut5_odd, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_orig_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_fivepcf,
        d_weight5pcf, f_alm, f_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_odd, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//* ==== ADD TO POWER 6 METHODS ===== *//
//6PCF kernels
//We have main (1) and orig(2) kernels for each of 3 precision modes
//run main kernel gpu == 1
void gpu_add_to_power6(double *d_sixpcf, double *d_weight6pcf,
        int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
        int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
        int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
        int *lut6_k, int *lut6_l, int *lut6_m,
        double wp, int nb, int nlm, int nouter, int ninner, int nell6) {
  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart6*nb*nlm;
  pstart6++;

if (count == 2) {
count++;
std::cout << "Threads6 = " << threads << " Nouter = " << nouter << " Ninner = " << ninner << std::endl;
}

  add_to_power6_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_sixpcf,
        d_weight6pcf, d_alm, d_almconj, lut6_l1, lut6_l2, lut6_l12, lut6_l3,
	lut6_l123, lut6_l4, lut6_l5, lut6_odd, lut6_n, lut6_zeta, lut6_i,
	lut6_j, lut6_k, lut6_l, lut6_m,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //gpu_print_cuda_error();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//float version of main kernel
void gpu_add_to_power6_float(float *d_sixpcf, float *d_weight6pcf,
        int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
        int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
        int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
        int *lut6_k, int *lut6_l, int *lut6_m,
        float wp, int nb, int nlm, int nouter, int ninner, int nell6) {
  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart6*nb*nlm;
  pstart6++;

  add_to_power6_kernel_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_sixpcf,
        d_weight6pcf, f_alm, f_almconj, lut6_l1, lut6_l2, lut6_l12, lut6_l3,
        lut6_l123, lut6_l4, lut6_l5, lut6_odd, lut6_n, lut6_zeta, lut6_i,
        lut6_j, lut6_k, lut6_l, lut6_m,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//mixed precision
void gpu_add_to_power6_mixed(double *d_sixpcf, double *d_weight6pcf,
        int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
        int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
        int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
        int *lut6_k, int *lut6_l, int *lut6_m,
        float wp, int nb, int nlm, int nouter, int ninner, int nell6) {
  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  //calculate index of d_alm for this particle
  int almidx = pstart6*nb*nlm;
  pstart6++;

  add_to_power6_kernel_mixed<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_sixpcf,
        d_weight6pcf, f_alm, f_almconj, lut6_l1, lut6_l2, lut6_l12, lut6_l3,
        lut6_l123, lut6_l4, lut6_l5, lut6_odd, lut6_n, lut6_zeta, lut6_i,
        lut6_j, lut6_k, lut6_l, lut6_m,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}


// ======================================================= /
//  ALL MEMORY FUNCTIONS ARE HERE                          /
// ======================================================= /
//allocate LUTs used in all kernels
//3PCF LUTs


void gpu_allocate_luts3(int **p_lut3_i, int **p_lut3_j, int **p_lut3_ct, int nouter) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut3_i), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut3_j), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut3_ct), nouter*sizeof(int));
}

void gpu_allocate_threepcf(double **p_threepcf, double *threepcf, int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_threepcf), size*sizeof(double));
  double *d_threepcf = *(p_threepcf);
  for (int i = 0; i < size; i++) d_threepcf[i] = threepcf[i];
}

void gpu_allocate_weight3pcf(double **p_weight3pcf, double *weight3pcf, int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_weight3pcf), size*sizeof(double));
  double *d_weight3pcf = *(p_weight3pcf);
  for (int i = 0; i < size; i++) d_weight3pcf[i] = weight3pcf[i];
}

void copy_threepcf(double **p_threepcf, double *threepcf, int size) {
  double *d_threepcf = *(p_threepcf);
  for (int i = 0; i < size; i++) threepcf[i] = d_threepcf[i];
}

void gpu_allocate_threepcf(float **p_threepcf, double *threepcf, int size) {
  cudaMallocManaged(&(*p_threepcf), size*sizeof(float));
  float *f_threepcf = *(p_threepcf);
  for (int i = 0; i < size; i++) f_threepcf[i] = (float)threepcf[i];
}

void gpu_allocate_weight3pcf(float **p_weight3pcf, double *weight3pcf, int size) {
  cudaMallocManaged(&(*p_weight3pcf), size*sizeof(float));
  float *f_weight3pcf = *(p_weight3pcf);
  for (int i = 0; i < size; i++) f_weight3pcf[i] = (float)weight3pcf[i];
}

void copy_threepcf(float **p_threepcf, double *threepcf, int size) {
  float *f_threepcf = *(p_threepcf);
  for (int i = 0; i < size; i++) threepcf[i] = (double)f_threepcf[i];
}

//* ==== FREE MEMORY 3 ==== *//

void gpu_free_luts3(int *lut3_i, int *lut3_j, int *lut3_ct) {
  cudaFree(lut3_i);
  cudaFree(lut3_j);
  cudaFree(lut3_ct);
}

void gpu_free_memory3(double *threepcf, double *weight3pcf) {
  cudaFree(threepcf);
  cudaFree(weight3pcf);
}

void gpu_free_memory3(float *threepcf, float *weight3pcf) {
  cudaFree(threepcf);
  cudaFree(weight3pcf);
}

// ======================================================= /
//DISCONNECTED 4PCF LUTs
void gpu_allocate_luts_discon1(int **p_lut_discon_ell, int **p_lut_discon_mm,
        int size) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut_discon_ell), size*sizeof(int));
  cudaMallocManaged(&(*p_lut_discon_mm), size*sizeof(int));
}

void gpu_allocate_discon1(double **p_discon1_r, double **p_discon1_i,
        std::complex<double> *discon1, int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_discon1_r), size*sizeof(double));
  cudaMallocManaged(&(*p_discon1_i), size*sizeof(double));
  double *d_discon1_r = *(p_discon1_r);
  double *d_discon1_i = *(p_discon1_i);
  for (int i = 0; i < size; i++) {
    d_discon1_r[i] = discon1[i].real();
    d_discon1_i[i] = discon1[i].imag();
  }
}

void gpu_allocate_weightdiscon(double **p_weightdiscon, double *weightdiscon,
	int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_weightdiscon), size*sizeof(double));
  double *d_weightdiscon = *(p_weightdiscon);
  for (int i = 0; i < size; i++) d_weightdiscon[i] = weightdiscon[i];
}

void copy_discon1(double **p_discon1_r, double **p_discon1_i,
        std::complex<double> *discon1, int size) {
  double *d_discon1_r = *(p_discon1_r);
  double *d_discon1_i = *(p_discon1_i);
  for (int i = 0; i < size; i++) {
    discon1[i] = {d_discon1_r[i], d_discon1_i[i]};
  }
}

void gpu_allocate_discon1(float **p_discon1_r, float **p_discon1_i,
        std::complex<double> *discon1, int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_discon1_r), size*sizeof(float));
  cudaMallocManaged(&(*p_discon1_i), size*sizeof(float));
  float *f_discon1_r = *(p_discon1_r);
  float *f_discon1_i = *(p_discon1_i);
  for (int i = 0; i < size; i++) {
    f_discon1_r[i] = (float)discon1[i].real();
    f_discon1_i[i] = (float)discon1[i].imag();
  }
}

void gpu_allocate_weightdiscon(float **p_weightdiscon, double *weightdiscon,
	int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_weightdiscon), size*sizeof(float));
  float *f_weightdiscon = *(p_weightdiscon);
  for (int i = 0; i < size; i++) f_weightdiscon[i] = (float)weightdiscon[i];
}

void copy_discon1(float **p_discon1_r, float **p_discon1_i,
        std::complex<double> *discon1, int size) {
  float *f_discon1_r = *(p_discon1_r);
  float *f_discon1_i = *(p_discon1_i);
  for (int i = 0; i < size; i++) {
    discon1[i] = {f_discon1_r[i], f_discon1_i[i]};
  }
}

//DISCON2 term
void gpu_allocate_luts_discon2(int **p_lut_discon_ell1,
        int **p_lut_discon_ell2, int **p_lut_discon_mm1,
        int **p_lut_discon_mm2, int size) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut_discon_ell1), size*sizeof(int));
  cudaMallocManaged(&(*p_lut_discon_ell2), size*sizeof(int));
  cudaMallocManaged(&(*p_lut_discon_mm1), size*sizeof(int));
  cudaMallocManaged(&(*p_lut_discon_mm2), size*sizeof(int));
}

void gpu_allocate_luts_discon2_inner(int **p_lut_discon_i,
        int **p_lut_discon_j, int size) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut_discon_i), size*sizeof(int));
  cudaMallocManaged(&(*p_lut_discon_j), size*sizeof(int));
}

void gpu_allocate_discon2(double **p_discon2_r, double **p_discon2_i,
        std::complex<double> *discon2, int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_discon2_r), size*sizeof(double));
  cudaMallocManaged(&(*p_discon2_i), size*sizeof(double));
  double *d_discon2_r = *(p_discon2_r);
  double *d_discon2_i = *(p_discon2_i);
  for (int i = 0; i < size; i++) {
    d_discon2_r[i] = discon2[i].real();
    d_discon2_i[i] = discon2[i].imag();
  }
}

void copy_discon2(double **p_discon2_r, double **p_discon2_i,
        std::complex<double> *discon2, int size) {
  double *d_discon2_r = *(p_discon2_r);
  double *d_discon2_i = *(p_discon2_i);
  for (int i = 0; i < size; i++) {
    discon2[i] = {d_discon2_r[i], d_discon2_i[i]};
  }
}

void gpu_allocate_discon2(float **p_discon2_r, float **p_discon2_i,
        std::complex<double> *discon2, int size) {
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_discon2_r), size*sizeof(float));
  cudaMallocManaged(&(*p_discon2_i), size*sizeof(float));
  float *f_discon2_r = *(p_discon2_r);
  float *f_discon2_i = *(p_discon2_i);
  for (int i = 0; i < size; i++) {
    f_discon2_r[i] = (float)discon2[i].real();
    f_discon2_i[i] = (float)discon2[i].imag();
  }
}

void copy_discon2(float **p_discon2_r, float **p_discon2_i,
        std::complex<double> *discon2, int size) {
  float *f_discon2_r = *(p_discon2_r);
  float *f_discon2_i = *(p_discon2_i);
  for (int i = 0; i < size; i++) {
    discon2[i] = {f_discon2_r[i], f_discon2_i[i]};
  }
}



// ======================================================= /

//* ==== FREE MEMORY DISCONNECTED ==== *//

void gpu_free_luts_discon1(int *lut_discon_ell, int *lut_discon_mm) {
  cudaFree(lut_discon_ell);
  cudaFree(lut_discon_mm);
}

void gpu_free_memory_discon1(double *d_discon1_r, double *d_discon1_i,
        double *weightdiscon) {
  cudaFree(d_discon1_r);
  cudaFree(d_discon1_i);
  cudaFree(weightdiscon);
}

void gpu_free_memory_discon1(float *f_discon1_r, float *f_discon1_i,
        float *weightdiscon) {
  cudaFree(f_discon1_r);
  cudaFree(f_discon1_i);
  cudaFree(weightdiscon);
}

//DISCON2 term
void gpu_free_luts_discon2(int *lut_discon_ell1, int *lut_discon_ell2,
        int *lut_discon_mm1, int *lut_discon_mm2, int *lut_discon_i,
	int *lut_discon_j) {
  cudaFree(lut_discon_ell1);
  cudaFree(lut_discon_ell2);
  cudaFree(lut_discon_mm1);
  cudaFree(lut_discon_mm2);
  cudaFree(lut_discon_i);
  cudaFree(lut_discon_j);
}


// ======================================================= /
//4PCF LUTs

void gpu_allocate_luts4(int **p_lut4_l1, int **p_lut4_l2, int **p_lut4_l3,
	bool **p_lut4_odd, int **p_lut4_n,
	int **p_lut4_zeta, int **p_lut4_i, int **p_lut4_j, int **p_lut4_k,
        int nouter, int ninner) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut4_l1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut4_l2), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut4_l3), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut4_odd), nouter*sizeof(bool));
  cudaMallocManaged(&(*p_lut4_n), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut4_zeta), nouter*sizeof(int));

  cudaMallocManaged(&(*p_lut4_i), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut4_j), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut4_k), ninner*sizeof(int));
}

void gpu_allocate_m_luts4(int **p_lut4_m1, int **p_lut4_m2, int nouter) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut4_m1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut4_m2), nouter*sizeof(int));
}

void gpu_allocate_fourpcf(double **p_fourpcf, double *fourpcf, int size) {
  //cudaMalloc(&(*p_fourpcf), size*sizeof(double));
  //cudaMemcpy((*p_fourpcf), fourpcf, size, cudaMemcpyHostToDevice);
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_fourpcf), size*sizeof(double));
  double *d_fourpcf = *(p_fourpcf);
  for (int i = 0; i < size; i++) d_fourpcf[i] = fourpcf[i];
}

void gpu_allocate_weight4pcf(double **p_weight4pcf, double *weight4pcf, int size) {
  //cudaMalloc(&(*p_weight4pcf), size*sizeof(double));
  //cudaMemcpy((*p_weight4pcf), weight4pcf, size, cudaMemcpyHostToDevice);
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_weight4pcf), size*sizeof(double));
  double *d_weight4pcf = *(p_weight4pcf);
  for (int i = 0; i < size; i++) d_weight4pcf[i] = weight4pcf[i];
}

void copy_fourpcf(double **p_fourpcf, double *fourpcf, int size) {
  cudaMemcpy(fourpcf, (*p_fourpcf), size*sizeof(double), cudaMemcpyDeviceToHost);
}

void gpu_allocate_fourpcf(float **p_fourpcf, double *fourpcf, int size) {
  cudaMallocManaged(&(*p_fourpcf), size*sizeof(float));
  float *f_fourpcf = *(p_fourpcf);
  for (int i = 0; i < size; i++) f_fourpcf[i] = (float)fourpcf[i];
}

void gpu_allocate_weight4pcf(float **p_weight4pcf, double *weight4pcf, int size) {
  cudaMallocManaged(&(*p_weight4pcf), size*sizeof(float));
  float *f_weight4pcf = *(p_weight4pcf);
  for (int i = 0; i < size; i++) f_weight4pcf[i] = (float)weight4pcf[i];
}

void copy_fourpcf(float **p_fourpcf, double *fourpcf, int size) {
  float *f_fourpcf = *(p_fourpcf);
  for (int i = 0; i < size; i++) fourpcf[i] = (double)f_fourpcf[i];
}

//* ==== FREE MEMORY 4 ==== *//

void gpu_free_luts4(int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
	int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k) {
  cudaFree(lut4_l1);
  cudaFree(lut4_l2);
  cudaFree(lut4_l3);
  cudaFree(lut4_odd);
  cudaFree(lut4_n);
  cudaFree(lut4_zeta);
  cudaFree(lut4_i);
  cudaFree(lut4_j);
  cudaFree(lut4_k);
}

void gpu_free_memory4(double *fourpcf, double *weight4pcf) {
  cudaFree(fourpcf);
  cudaFree(weight4pcf);
}

void gpu_free_memory4(float *fourpcf, float *weight4pcf) {
  cudaFree(fourpcf);
  cudaFree(weight4pcf);
}

void gpu_free_memory_m4(int *lut4_m1, int *lut4_m2) {
  cudaFree(lut4_m1);
  cudaFree(lut4_m2);
}

// ======================================================= /
//5PCF LUTs

void gpu_allocate_luts(int **p_lut5_l1, int **p_lut5_l2, int **p_lut5_l12,
	int **p_lut5_l3, int **p_lut5_l4, bool **p_lut5_odd, int **p_lut5_n,
        int **p_lut5_zeta, int **p_lut5_i, int **p_lut5_j, int **p_lut5_k,
	int **p_lut5_l, int nouter, int ninner) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut5_l1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l2), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l12), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l3), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l4), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_odd), nouter*sizeof(bool));
  cudaMallocManaged(&(*p_lut5_n), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_zeta), nouter*sizeof(int));

  cudaMallocManaged(&(*p_lut5_i), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut5_j), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut5_k), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l), ninner*sizeof(int));
}

void gpu_allocate_m_luts(int **p_lut5_m1, int **p_lut5_m2, int **p_lut5_m3,
	int nouter) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut5_m1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_m2), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_m3), nouter*sizeof(int));
}

void gpu_allocate_fivepcf(double **p_fivepcf, double *fivepcf, int size) {
  //*p_fivepcf = (double *)malloc(sizeof(double)*size);
  //cudaMalloc(&(*p_fivepcf), size*sizeof(double));
  //cudaMemcpy((*p_fivepcf), fivepcf, size, cudaMemcpyHostToDevice);
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_fivepcf), size*sizeof(double));
  double *d_fivepcf = *(p_fivepcf);
  for (int i = 0; i < size; i++) d_fivepcf[i] = fivepcf[i];
}

void gpu_allocate_weight5pcf(double **p_weight5pcf, double *weight5pcf, int size) {
  //cudaMalloc(&(*p_weight5pcf), size*sizeof(double));
  //cudaMemcpy((*p_weight5pcf), weight5pcf, size, cudaMemcpyHostToDevice);
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_weight5pcf), size*sizeof(double));
  double *d_weight5pcf = *(p_weight5pcf);
  for (int i = 0; i < size; i++) d_weight5pcf[i] = weight5pcf[i];
}

void copy_fivepcf(double **p_fivepcf, double *fivepcf, int size) {
  cudaMemcpy(fivepcf, (*p_fivepcf), size*sizeof(double), cudaMemcpyDeviceToHost);
}

void gpu_allocate_fivepcf(float **p_fivepcf, double *fivepcf, int size) {
  cudaMallocManaged(&(*p_fivepcf), size*sizeof(float));
  float *f_fivepcf = *(p_fivepcf);
  for (int i = 0; i < size; i++) f_fivepcf[i] = (float)fivepcf[i];
}

void gpu_allocate_weight5pcf(float **p_weight5pcf, double *weight5pcf, int size) {
  cudaMallocManaged(&(*p_weight5pcf), size*sizeof(float));
  float *f_weight5pcf = *(p_weight5pcf);
  for (int i = 0; i < size; i++) f_weight5pcf[i] = (float)weight5pcf[i];
}

void copy_fivepcf(float **p_fivepcf, double *fivepcf, int size) {
  float *f_fivepcf = *(p_fivepcf);
  for (int i = 0; i < size; i++) fivepcf[i] = (double)f_fivepcf[i];
}

//* ==== FREE MEMORY 5 ==== *//

void gpu_free_luts(int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, bool *lut5_odd, int *lut5_n, int *lut5_zeta, int *lut5_i,
	int *lut5_j, int *lut5_k, int *lut5_l) {
  cudaFree(lut5_l1);
  cudaFree(lut5_l2);
  cudaFree(lut5_l12);
  cudaFree(lut5_l3);
  cudaFree(lut5_l4);
  cudaFree(lut5_odd);
  cudaFree(lut5_n);
  cudaFree(lut5_zeta);
  cudaFree(lut5_i);
  cudaFree(lut5_j);
  cudaFree(lut5_k);
  cudaFree(lut5_l);
}

void gpu_free_memory(double *fivepcf, double *weight5pcf) {
  cudaFree(fivepcf);
  cudaFree(weight5pcf);
}

void gpu_free_memory(float *fivepcf, float *weight5pcf) {
  cudaFree(fivepcf);
  cudaFree(weight5pcf);
}

void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3) {
  cudaFree(lut5_m1);
  cudaFree(lut5_m2);
  cudaFree(lut5_m3);
}

// ======================================================= /
//6PCF LUTs

void gpu_allocate_luts6(int **p_lut6_l1, int **p_lut6_l2, int **p_lut6_l12,
        int **p_lut6_l3, int **p_lut6_l123, int **p_lut6_l4,
        int **p_lut6_l5, bool **p_lut6_odd, int **p_lut6_n,
        int **p_lut6_zeta, int **p_lut6_i, int **p_lut6_j, int **p_lut6_k,
        int **p_lut6_l, int **p_lut6_m, int nouter, int ninner) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut6_l1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_l2), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_l12), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_l3), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_l123), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_l4), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_l5), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_odd), nouter*sizeof(bool));
  cudaMallocManaged(&(*p_lut6_n), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut6_zeta), nouter*sizeof(int));

  cudaMallocManaged(&(*p_lut6_i), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut6_j), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut6_k), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut6_l), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut6_m), ninner*sizeof(int));
}

void gpu_allocate_sixpcf(double **p_sixpcf, double *sixpcf, int size) {
  //*p_sixpcf = (double *)malloc(sizeof(double)*size);
  //cudaMalloc(&(*p_sixpcf), size*sizeof(double));
  //cudaMemcpy((*p_sixpcf), sixpcf, size, cudaMemcpyHostToDevice);
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_sixpcf), size*sizeof(double));
  double *d_sixpcf = *(p_sixpcf);
  for (int i = 0; i < size; i++) d_sixpcf[i] = sixpcf[i];
}

void gpu_allocate_weight6pcf(double **p_weight6pcf, double *weight6pcf, int size) {
  //cudaMalloc(&(*p_weight6pcf), size*sizeof(double));
  //cudaMemcpy((*p_weight6pcf), weight6pcf, size, cudaMemcpyHostToDevice);
  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
  cudaMallocManaged(&(*p_weight6pcf), size*sizeof(double));
  double *d_weight6pcf = *(p_weight6pcf);
  for (int i = 0; i < size; i++) d_weight6pcf[i] = weight6pcf[i];
}

void copy_sixpcf(double **p_sixpcf, double *sixpcf, int size) {
  cudaMemcpy(sixpcf, (*p_sixpcf), size*sizeof(double), cudaMemcpyDeviceToHost);
}

void gpu_allocate_sixpcf(float **p_sixpcf, double *sixpcf, int size) {
  cudaMallocManaged(&(*p_sixpcf), size*sizeof(float));
  float *f_sixpcf = *(p_sixpcf);
  for (int i = 0; i < size; i++) f_sixpcf[i] = (float)sixpcf[i];
}

void gpu_allocate_weight6pcf(float **p_weight6pcf, double *weight6pcf, int size) {
  cudaMallocManaged(&(*p_weight6pcf), size*sizeof(float));
  float *f_weight6pcf = *(p_weight6pcf);
  for (int i = 0; i < size; i++) f_weight6pcf[i] = (float)weight6pcf[i];
}

void copy_sixpcf(float **p_sixpcf, double *sixpcf, int size) {
  float *f_sixpcf = *(p_sixpcf);
  for (int i = 0; i < size; i++) sixpcf[i] = (double)f_sixpcf[i];
}

//* ==== FREE MEMORY 6 ==== *//

void gpu_free_luts6(int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
        int *lut6_l123, int *lut6_l4, int *lut6_l5, bool *lut6_odd,
        int *lut6_n, int *lut6_zeta, int *lut6_i, int *lut6_j,
        int *lut6_k, int *lut6_l, int *lut6_m) {
  cudaFree(lut6_l1);
  cudaFree(lut6_l2);
  cudaFree(lut6_l12);
  cudaFree(lut6_l3);
  cudaFree(lut6_l123);
  cudaFree(lut6_l4);
  cudaFree(lut6_l5);
  cudaFree(lut6_odd);
  cudaFree(lut6_n);
  cudaFree(lut6_zeta);
  cudaFree(lut6_i);
  cudaFree(lut6_j);
  cudaFree(lut6_k);
  cudaFree(lut6_l);
  cudaFree(lut6_m);
}

void gpu_free_memory6(double *sixpcf, double *weight6pcf) {
  cudaFree(sixpcf);
  cudaFree(weight6pcf);
}

void gpu_free_memory6(float *sixpcf, float *weight6pcf) {
  cudaFree(sixpcf);
  cudaFree(weight6pcf);
}

// ======================================================= /
//  ALL ALM FUNCTIONS ARE HERE                             /
// ======================================================= /

//* ==== ALLOCATE ALMS ==== *//

void gpu_allocate_alms(int np, int nb, int nlm, bool isDouble) {
  //d_alm and d_almconj are already declared at top of gpufuncs.cu
  //so are f_alm and f_almconj.  Need to select based on kernel.
  if (isDouble) {
    cudaMallocManaged(&d_alm, np*nb*nlm*sizeof(thrust::complex<double>));
    cudaMallocManaged(&d_almconj, np*nb*nlm*sizeof(thrust::complex<double>));
  } else {
    cudaMallocManaged(&f_alm, np*nb*nlm*sizeof(thrust::complex<float>));
    cudaMallocManaged(&f_almconj, np*nb*nlm*sizeof(thrust::complex<float>));
  }
}

void gpu_compute_alms(int *map, double *m, int nbin, int nlm, int maxp,
	int order, int mapdim, int nmult) {

  pstart5 = 0; //reset pstart5 each time alms are calculated
  pstart = 0; //ALSO pstart (4)

  int *d_map;
  size_t size_map = sizeof(int)*mapdim*mapdim*mapdim;
  cudaMallocManaged(&d_map, size_map);
  int n = 0;
  //copy map to GPU memory
  for (int a = 0; a < mapdim; a++) {
    for (int b = 0; b < mapdim; b++) {
      for (int c = 0; c < mapdim; c++) {
	d_map[n] = map[n];
	n++;
      }
    }
  }

  // Invoke kernel
  long threads = maxp*nbin;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  compute_alms<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_alm, d_almconj, d_map, m,
	nbin, nlm, maxp, order, mapdim, nmult);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}

void gpu_compute_alms_float(int *map, double *m, int nbin, int nlm, int maxp,
	int order, int mapdim, int nmult) {

  pstart5 = 0; //reset pstart5 each time alms are calculated

  int *d_map;
  size_t size_map = sizeof(int)*mapdim*mapdim*mapdim;
  cudaMallocManaged(&d_map, size_map);
  int n = 0;
  //copy map to GPU memory
  for (int a = 0; a < mapdim; a++) {
    for (int b = 0; b < mapdim; b++) {
      for (int c = 0; c < mapdim; c++) {
        d_map[n] = map[n];
        n++;
      }
    }
  }

  // Invoke kernel
  long threads = maxp*nbin;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  compute_alms_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(f_alm, f_almconj,
	d_map, m, nbin, nlm, maxp, order, mapdim, nmult);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}

//* ==== FREE ALMS ==== *//

void gpu_free_memory_alms(bool isDouble) {
  //free d_alm and d_almconj at end of run
  if (isDouble) {
    cudaFree(d_alm);
    cudaFree(d_almconj);
  } else {
    cudaFree(f_alm);
    cudaFree(f_almconj);
  }
}

// ======================================================= /
//  PAIRS AND MULTIPOLES ACCUMULATION                      /
// ======================================================= /

//memory operation

void gpu_allocate_multipoles(double **p_msave, int **p_csave,
        int **p_pnum, int **p_spnum, int **p_snp, int **p_sc,
        int nmult, int nbin, int np, int nmax) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_msave), nmult*nbin*np*sizeof(double));
  cudaMallocManaged(&(*p_csave), np*nbin*sizeof(int));
  cudaMallocManaged(&(*p_pnum), nmax*sizeof(int));
  cudaMallocManaged(&(*p_spnum), nmax*sizeof(int));
  cudaMallocManaged(&(*p_snp), nmax*sizeof(int));
  cudaMallocManaged(&(*p_sc), nmax*sizeof(int));
}

void gpu_allocate_multipoles_fast(double **p_msave, int **p_csave,
        int **p_start_list, int **p_np_list, int **p_cellnums,
        int nmult, int nbin, int np, int maxp, int nmax, int nc) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_msave), nmult*nbin*maxp*sizeof(double));
  cudaMallocManaged(&(*p_csave), maxp*nbin*sizeof(int));
  cudaMallocManaged(&(*p_start_list), nc*sizeof(int));
  cudaMallocManaged(&(*p_np_list), nc*sizeof(int));
  cudaMallocManaged(&(*p_cellnums), np*sizeof(int));
}

void gpu_allocate_particle_arrays(double **p_posx, double **p_posy, double **p_posz, double **p_weights, int np) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_posx), np*sizeof(double));
  cudaMallocManaged(&(*p_posy), np*sizeof(double));
  cudaMallocManaged(&(*p_posz), np*sizeof(double));
  cudaMallocManaged(&(*p_weights), np*sizeof(double));
}

void gpu_allocate_pair_arrays(double **p_x0i, double **p_x2i, int nbin) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_x0i), nbin*sizeof(double));
  cudaMallocManaged(&(*p_x2i), nbin*sizeof(double));

}

void gpu_allocate_periodic(int **p_delta_x, int **p_delta_y, int ** p_delta_z, int nmax) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_delta_x), nmax*sizeof(int));
  cudaMallocManaged(&(*p_delta_y), nmax*sizeof(int));
  cudaMallocManaged(&(*p_delta_z), nmax*sizeof(int));
}

//* ==== FREE MULTIPOLES AND PARTICLES ==== *//

void free_gpu_multipole_arrays(double *msave, int *csave,
        int *pnum, int *spnum, int *snp, int *sc,
	double *posx, double *posy, double *posz,
	double *weights, double *x0i, double *x2i) {
  cudaFree(msave);
  cudaFree(csave);
  cudaFree(pnum);
  cudaFree(spnum);
  cudaFree(snp);
  cudaFree(sc);
  cudaFree(posx);
  cudaFree(posy);
  cudaFree(posz);
  cudaFree(weights);
  cudaFree(x0i);
  cudaFree(x2i);
}

void free_gpu_periodic_arrays(int *delta_x, int *delta_y, int *delta_z) {
  cudaFree(delta_x);
  cudaFree(delta_y);
  cudaFree(delta_z);
}

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
	int n, int nbin, float rmin, float rmax, bool shared, bool gpufloat) {
  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  if (gpufloat) {
    if (shared) {
      add_pairs_only_kernel_float_shared<<<blocksPerGrid, THREADS_PER_BLOCK>>>
	(posx, posy, posz, w, pnum, spnum, snp, sc, x0i, x2i, n, nbin,
	rmin, rmax, rmin2, rmax2);
    } else {
      add_pairs_only_kernel_float<<<blocksPerGrid, THREADS_PER_BLOCK>>>(posx,
	posy, posz, w, pnum, spnum, snp, sc, x0i, x2i, n, nbin, rmin, rmax,
	rmin2, rmax2);
    }
  } else {
    if (shared) {
      add_pairs_only_kernel_shared<<<blocksPerGrid, THREADS_PER_BLOCK>>>(posx,
	posy, posz, w, pnum, spnum, snp, sc, x0i, x2i, n, nbin, rmin, rmax,
	rmin2, rmax2);
    } else {
      add_pairs_only_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(posx,
	posy, posz, w, pnum, spnum, snp, sc, x0i, x2i, n, nbin, rmin, rmax,
	rmin2, rmax2);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}

void gpu_add_pairs_only_periodic(double *posx, double *posy, double *posz,
	double *w, int *pnum, int *spnum, int *snp, int *sc, double *x0i,
	double *x2i, int *delta_x, int *delta_y, int *delta_z, int n,
	int nbin, float rmin, float rmax, double cellsize, bool shared,
	bool gpufloat) {
  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  if (gpufloat) {
    if (shared) {
      add_pairs_only_periodic_kernel_float_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, pnum, spnum, snp, sc,
	x0i, x2i, delta_x, delta_y, delta_z, n, nbin, rmin, rmax, rmin2,
	rmax2, (float)cellsize);
    } else {
      add_pairs_only_periodic_kernel_float<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, pnum, spnum, snp, sc,
	x0i, x2i, delta_x, delta_y, delta_z, n, nbin, rmin, rmax, rmin2,
	rmax2, (float)cellsize);
    }
  } else {
    if (shared) {
      add_pairs_only_periodic_kernel_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, pnum, spnum, snp, sc,
	x0i, x2i, delta_x, delta_y, delta_z, n, nbin, rmin, rmax, rmin2,
	rmax2, cellsize);
    } else {
      add_pairs_only_periodic_kernel<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, pnum, spnum, snp, sc,
	x0i, x2i, delta_x, delta_y, delta_z, n, nbin, rmin, rmax, rmin2,
	rmax2, cellsize);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}

void gpu_add_pairs_only_fast(double *posx, double *posy, double *posz,
	double *w, int *start_list, int *np_list, int *cellnums, double *x0i,
        double *x2i, int n, int nbin, int order,
        int nmult, float rmin, float rmax, int nside_cuboid_x,
        int nside_cubiod_y, int nside_cuboid_z, int ncells, int maxsep,
        int pstart, bool shared, bool gpufloat) {

  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int3 nside_cuboid = make_int3(nside_cuboid_x, nside_cubiod_y, nside_cuboid_z);
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  if (gpufloat) {
    if (shared) {
      add_pairs_only_kernel_fast_float_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
	cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    } else {
      add_pairs_only_kernel_fast_float<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
	cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    }
  } else {
    if (shared) {
      add_pairs_only_kernel_fast_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
	cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    } else {
      add_pairs_only_kernel_fast<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
	cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}

void gpu_add_pairs_only_periodic_fast(double *posx, double *posy, double *posz,
        double *w, int *start_list, int *np_list, int *cellnums, double *x0i,
        double *x2i, int n, int nbin, int order, int nmult, float rmin,
	float rmax, int nside_cuboid_x, int nside_cubiod_y,
	int nside_cuboid_z, int ncells, int maxsep, int pstart,
	double cellsize, bool shared, bool gpufloat) { 

  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int3 nside_cuboid = make_int3(nside_cuboid_x, nside_cubiod_y, nside_cuboid_z);
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  if (gpufloat) {
    if (shared) {
      add_pairs_only_periodic_kernel_fast_float_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
	cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart, (float)cellsize);
    } else {
      add_pairs_only_periodic_kernel_fast_float<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
        cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart, (float)cellsize);
    }
  } else {
    if (shared) {
      add_pairs_only_periodic_kernel_fast_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
        cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart, cellsize);
    } else {
      add_pairs_only_periodic_kernel_fast<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(posx, posy, posz, w, start_list, np_list,
        cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart, cellsize);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}


//PAIRS AND MULTIPOLES

void gpu_add_pairs_and_multipoles(double *m, double *posx, double *posy,
        double *posz, double *w, int *ct, int *pnum, int *spnum,
	int *snp, int *sc, double *x0i, double *x2i, int n, int nbin,
	int order, int nmult, float rmin, float rmax, int pstart,
	bool shared, bool gpufloat) {
  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  std::clog << " NTHREADS " << threads << " NBIN " << nbin << std::endl;
  std::clog << "N " << n << " BLOCKS " << blocksPerGrid << std::endl;

  if (gpufloat) {
    if (shared) {
      add_pairs_and_multipoles_kernel_float_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum,
	snp, sc, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2,
	rmax2, pstart);
    } else {
      add_pairs_and_multipoles_kernel_float<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum,
        snp, sc, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2,
        rmax2, pstart);
    }
  } else {
    if (shared) {
      add_pairs_and_multipoles_kernel_shared<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum,
        snp, sc, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2,
        rmax2, pstart); 
    } else {
      add_pairs_and_multipoles_kernel<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum,
        snp, sc, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2,
        rmax2, pstart);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}

void gpu_add_pairs_and_multipoles_fast(double *m, double *posx, double *posy,
        double *posz, double *w, int *ct, int *start_list, int *np_list,
	int *cellnums, double *x0i, double *x2i, int n, int nbin, int order,
	int nmult, float rmin, float rmax, int nside_cuboid_x,
	int nside_cubiod_y, int nside_cuboid_z, int ncells, int maxsep,
	int pstart, bool shared, bool gpufloat) {

  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int3 nside_cuboid = make_int3(nside_cuboid_x, nside_cubiod_y, nside_cuboid_z);
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
  std::clog << " NTHREADS " << threads << " NBIN " << nbin << std::endl;
  std::clog << "N " << n << " BLOCKS " << blocksPerGrid << std::endl;

  if (gpufloat) {
    if (shared) {
      add_pairs_and_multipoles_kernel_fast_float_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, start_list, np_list,
	cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    } else {
      add_pairs_and_multipoles_kernel_fast_float<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, start_list, np_list,
        cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    }
  } else {
    if (shared) {
      add_pairs_and_multipoles_kernel_fast_shared<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, start_list, np_list,
        cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    } else {
      add_pairs_and_multipoles_kernel_fast<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, start_list, np_list,
        cellnums, x0i, x2i, n, nbin, order, nmult, rmin, rmax, rmin2, rmax2,
        nside_cuboid, ncells, maxsep, pstart);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
}

void gpu_add_pairs_and_multipoles_periodic(double *m, double *posx,
        double *posy, double *posz, double *w, int *ct, int *pnum, int *spnum,
        int *snp, int *sc, double *x0i, double *x2i, int *delta_x,
        int *delta_y, int *delta_z, int n, int nbin, int order, int nmult,
        float rmin, float rmax, int pstart, double cellsize,
	bool shared, bool gpufloat) {
  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  if (gpufloat) {
    if (shared) {
      add_pairs_and_multipoles_periodic_kernel_float_shared<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum, snp,
	sc, x0i, x2i, delta_x, delta_y, delta_z, n, nbin, order, nmult,
	rmin, rmax, rmin2, rmax2, pstart, (float)cellsize);
    } else {
      add_pairs_and_multipoles_periodic_kernel_float<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum, snp,
	sc, x0i, x2i, delta_x, delta_y, delta_z, n, nbin, order, nmult,
	rmin, rmax, rmin2, rmax2, pstart, (float)cellsize);
    }
  } else {
    if (shared) {
      add_pairs_and_multipoles_periodic_kernel_shared<<<blocksPerGrid,
	THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum, snp,
	sc, x0i, x2i, delta_x, delta_y, delta_z, n, nbin, order, nmult,
	rmin, rmax, rmin2, rmax2, pstart, cellsize);
    } else {
      add_pairs_and_multipoles_periodic_kernel<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct, pnum, spnum, snp,
	sc, x0i, x2i, delta_x, delta_y, delta_z, n, nbin, order, nmult,
	rmin, rmax, rmin2, rmax2, pstart, cellsize);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //gpu_print_cuda_error();
}

void gpu_add_pairs_and_multipoles_periodic_fast(double *m, double *posx,
        double *posy, double *posz, double *w, int *ct, int *start_list,
        int *np_list, int *cellnums, double *x0i, double *x2i, int n,
	int nbin, int order, int nmult, float rmin, float rmax,
	int nside_cuboid_x, int nside_cubiod_y, int nside_cuboid_z,
	int ncells, int maxsep, int pstart, double cellsize,
	bool shared, bool gpufloat) {
  // Invoke kernel
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int3 nside_cuboid = make_int3(nside_cuboid_x, nside_cubiod_y, nside_cuboid_z);
  int blocksPerGrid = (threads+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

  if (gpufloat) {
    if (shared) {
      add_pairs_and_multipoles_periodic_kernel_fast_float_shared<<<
	blocksPerGrid, THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct,
	start_list, np_list, cellnums, x0i, x2i, n, nbin, order, nmult,
	rmin, rmax, rmin2, rmax2, nside_cuboid, ncells, maxsep, pstart,
	(float)cellsize);
    } else {
      add_pairs_and_multipoles_periodic_kernel_fast_float<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct,
        start_list, np_list, cellnums, x0i, x2i, n, nbin, order, nmult,
        rmin, rmax, rmin2, rmax2, nside_cuboid, ncells, maxsep, pstart,
        (float)cellsize);
    }
  } else {
    if (shared) {
      add_pairs_and_multipoles_periodic_kernel_fast_shared<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct,
        start_list, np_list, cellnums, x0i, x2i, n, nbin, order, nmult,
        rmin, rmax, rmin2, rmax2, nside_cuboid, ncells, maxsep, pstart,
        cellsize);
    } else {
      add_pairs_and_multipoles_periodic_kernel_fast<<<blocksPerGrid,
        THREADS_PER_BLOCK>>>(m, posx, posy, posz, w, ct,
        start_list, np_list, cellnums, x0i, x2i, n, nbin, order, nmult,
        rmin, rmax, rmin2, rmax2, nside_cuboid, ncells, maxsep, pstart,
        cellsize);
    }
  }

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
  //gpu_print_cuda_error();
}


void gpu_device_synchronize() {
  // Wait for GPU to finish before accessing on host
  //This does not need to be called after every kernel invocation,
  //but just before memory is accessed on host
  cudaDeviceSynchronize();
}

void gpu_print_cuda_error() {
/*
       size_t free_byte ;

        size_t total_byte ;

        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){

            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

            exit(1);

        }

double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
*/

cudaError_t err = cudaGetLastError();
printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
