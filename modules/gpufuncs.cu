#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include "gpufuncs.h"
#include <thrust/complex.h>
#include <cuComplex.h>

bool first = true;
int count = 0;

__global__ void test3_kernel(double *fivepcf, double *weight5pcf, double* alm_real,
        double* alm_imag, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nlm, int nouter, int ninner, int nl) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner *nl) return;
    i /= nl;
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
    int n = lut5_n[iouter]; //this is the starting n for this thread
    //inner loop indices
    int ii = lut5_i[iinner];
    int j = lut5_j[iinner];
    int k = lut5_k[iinner];
    int l = lut5_l[iinner];
    //alms
    double alm1w = 0;
    double alm2 = 0;
    double alm3 = 0;
    double alm1wi = 0;
    double alm2i = 0;
    double alm3i =0;
    double temp_r, temp_i;
    int m4, tmp_lm4;
    double delta;
    //now loop over ms on this thread
    // Iterate over all m1 (including negative)
    for(int m1=-l1; m1<=l1; m1++){
      // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
      if (m1 < 0) {
	alm1w = wp*alm_real[ii*nlm+tmp_l1-m1];
	alm1wi = -1*wp*alm_imag[ii*nlm+tmp_l1-m1]; //conjugate so multiply by -1
      } else {
	alm1w = wp*alm_real[ii*nlm+tmp_l1+m1];
	alm1wi = wp*alm_imag[ii*nlm+tmp_l1+m1];
      }
      //if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1].real(); else alm1w = wp*alm[ii*nlm+tmp_l1+m1].real();
      //if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
      //if (m1 < 0) alm1w = wp*ii+nlm-tmp_l1-m1; else alm1w = wp*ii+nlm+tmp_l1+m1;
      // Iterate over all m2 (including negative)
      for(int m2=-l2; m2<=l2; m2++){
        if(abs(m1+m2)>l12) continue; // m12 condition
        // Create temporary copy of a_l2m2, taking conjugate if necessary
        if (m2 < 0) {
          temp_r = alm_real[j*nlm+tmp_l2-m2];
          temp_i = alm_imag[j*nlm+tmp_l2-m2];
	  alm2 = alm1w*temp_r+alm1wi*temp_i; //conjugate cancels out sign
	  alm2i = -1*alm1w*temp_i+alm1wi*temp_r; //conjugate first term
        } else {
          temp_r = alm_real[j*nlm+tmp_l2+m2];
          temp_i = alm_imag[j*nlm+tmp_l2+m2];
	  alm2 = alm1w*temp_r-alm1wi*temp_i;
	  alm2i = alm1w*temp_i+alm1wi*temp_r;
        }
        //if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
        //if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2].real(); else alm2 = alm1w*alm[j*nlm+tmp_l2+m2].real();
        //if (m2 < 0) alm2 = alm1w*j+tmp_l2-m2; else alm2 = alm1w*j+tmp_l2+m2;
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
	  if (m3 < 0) {
	    temp_r = alm_real[k*nlm+tmp_l3-m3];
	    temp_i = alm_imag[k*nlm+tmp_l3-m3];
	    alm3 = alm2*temp_r+alm2i*temp_i; //conjugate cancels out sign
	    alm3i = -1*alm2*temp_i+alm2*temp_r; //conjugate first term
	  } else {
            temp_r = alm_real[k*nlm+tmp_l3+m3];
            temp_i = alm_imag[k*nlm+tmp_l3+m3];
	    alm3 = alm2*temp_r-alm2i*temp_i;
	    alm3i = alm2*temp_i+alm2i*temp_r;
	  }
          //if (m3 < 0) alm3 = alm2*almconj[k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[k*nlm+tmp_l3+m3];
          //if (m3 < 0) alm3 = alm2*almconj[k*nlm+tmp_l3-m3].real(); else alm3 = alm2*alm[k*nlm+tmp_l3+m3].real();
          //if (m3 < 0) alm3 = alm2*k+tmp_l3-m3; else alm3 = alm2*k+tmp_l3+m3;
          //calculate delta
          //delta = weight*(alm3*alm[l*nlm+tmp_lm4]).real();
          //delta = weight*(alm3*alm[l*nlm+tmp_lm4].real());
	  temp_r = alm_real[l*nlm+tmp_lm4];
	  temp_i = alm_imag[l*nlm+tmp_lm4];
	  delta = weight*(alm3*temp_r-alm3i*temp_i);
          //delta = weight*alm3*l*nlm+tmp_lm4;
          //delta = weight*ii;
          //add to this element
          pcf_element += delta;
        }
      }
    }
    fivepcf[bin_index] = pcf_element;
}

__global__ void test2_kernel(double *fivepcf, double *weight5pcf, thrust::complex<double>* alm,
        thrust::complex<double> *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nlm, int nouter, int ninner, int nl) {
    //thread index i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nouter * ninner *nl) return;
    i /= nl;
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
    for(int m1=-l1; m1<=l1; m1++){
      // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
      if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
      // Iterate over all m2 (including negative)
      for(int m2=-l2; m2<=l2; m2++){
        if(abs(m1+m2)>l12) continue; // m12 condition
        // Create temporary copy of a_l2m2, taking conjugate if necessary
        if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
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
          if (m3 < 0) alm3 = alm2*almconj[k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[k*nlm+tmp_l3+m3];
          //calculate delta
          delta = weight*(alm3*alm[l*nlm+tmp_lm4]).real();
          //add to this element
          pcf_element += delta;
        }
      }
    }
    //alm1w = wp*alm[ii*nlm+tmp_l1+l1];
    //delta = weight*(alm1w*alm[l*nlm+tmp_lm4]).real(); 
    //pcf_element += delta;
    fivepcf[bin_index] = pcf_element; 
}

__global__ void add_to_power5_kernel(double *fivepcf, double *weight5pcf, thrust::complex<double>* alm,
	thrust::complex<double> *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
    for(int m1=-l1; m1<=l1; m1++){
      // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
      if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
      // Iterate over all m2 (including negative)
      for(int m2=-l2; m2<=l2; m2++){
        if(abs(m1+m2)>l12) continue; // m12 condition
        // Create temporary copy of a_l2m2, taking conjugate if necessary
        if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
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
          if (m3 < 0) alm3 = alm2*almconj[k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[k*nlm+tmp_l3+m3];
	  //calculate delta
          delta = weight*(alm3*alm[l*nlm+tmp_lm4]).real();
	  //add to this element
	  pcf_element += delta;
        }
      }
    }
    fivepcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power5_kernel_orig(double *fivepcf, double *weight5pcf, thrust::complex<double>* alm,
	thrust::complex<double> *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n, int *lut5_zeta,
	int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        double wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
    if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
    int m4 = -m1-m2-m3;
    int tmp_lm4 = tmp_l4+m4;
    if (m3 < 0) alm3 = alm2*almconj[k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[k*nlm+tmp_l3+m3];
    double delta = weight*(alm3*alm[l*nlm+tmp_lm4]).real();
    atomicAdd(&fivepcf[bin_index], delta);
    //atomicAdd(&fivepcf[bin_index], m2);
}


void gpu_allocate_luts(int **p_lut5_l1, int **p_lut5_l2, int **p_lut5_l12, int **p_lut5_l3,
        int **p_lut5_l4, int **p_lut5_n,
        int **p_lut5_zeta, int **p_lut5_i, int **p_lut5_j, int **p_lut5_k, int **p_lut5_l,
        int nouter, int ninner) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut5_l1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l2), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l12), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l3), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l4), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_n), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_zeta), nouter*sizeof(int));

  cudaMallocManaged(&(*p_lut5_i), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut5_j), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut5_k), ninner*sizeof(int));
  cudaMallocManaged(&(*p_lut5_l), ninner*sizeof(int));
}

void gpu_allocate_m_luts(int **p_lut5_m1, int **p_lut5_m2, int **p_lut5_m3, int nouter) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut5_m1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_m2), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut5_m3), nouter*sizeof(int));
}

void gpu_allocate_fivepcf(double **p_fivepcf, double *fivepcf, int size) {
  //*p_fivepcf = (double *)malloc(sizeof(double)*size);
  cudaMalloc(&(*p_fivepcf), size*sizeof(double));
  cudaMemcpy((*p_fivepcf), fivepcf, size, cudaMemcpyHostToDevice);
}

void gpu_allocate_weight5pcf(double **p_weight5pcf, double *weight5pcf, int size) {
  cudaMalloc(&(*p_weight5pcf), size*sizeof(double));
  cudaMemcpy((*p_weight5pcf), weight5pcf, size, cudaMemcpyHostToDevice);
}

void gpu_free_memory(double *fivepcf, double *weight5pcf,
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l) {
  cudaFree(lut5_l1);
  cudaFree(lut5_l2);
  cudaFree(lut5_l12);
  cudaFree(lut5_l3);
  cudaFree(lut5_l4);
  cudaFree(lut5_n);
  cudaFree(lut5_zeta);
  cudaFree(lut5_i);
  cudaFree(lut5_j);
  cudaFree(lut5_k);
  cudaFree(lut5_l);
  cudaFree(fivepcf);
  cudaFree(weight5pcf);
}

void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3) {
  cudaFree(lut5_m1);
  cudaFree(lut5_m2);
  cudaFree(lut5_m3);
}


void gpu_add_to_power5(double *fivepcf, double *weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

/*
  int *d_lut5_l1, *d_lut5_l2, *d_lut5_l12, *d_lut5_l3;
  int *d_lut5_l4, *d_lut5_n;
  int *d_lut5_zeta, *d_lut5_i, *d_lut5_j, *d_lut5_k, *d_lut5_l;
*/

  double* d_fivepcf, *d_weight5pcf;
  thrust::complex<double>* d_alm, *d_almconj;
  //size_t size_outer = nouter * sizeof(int);
  //size_t size_inner = ninner * sizeof(int);
  size_t size_w = sizeof(double)*(norder+1)*(norder+1)*(norder+1)*(norder+1)*(2*norder+1)*(norder+1)*(norder+1)*(norder+1);
  size_t size_5 = sizeof(double)*nell5*ninner;

/*
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&d_lut5_l1, nouter*sizeof(int));
  cudaMalloc(&d_lut5_l2, nouter*sizeof(int));
  cudaMalloc(&d_lut5_l12, nouter*sizeof(int));
  cudaMalloc(&d_lut5_l3, nouter*sizeof(int));
  cudaMalloc(&d_lut5_l4, nouter*sizeof(int));
  cudaMalloc(&d_lut5_n, nouter*sizeof(int));
  cudaMalloc(&d_lut5_zeta, nouter*sizeof(int));

  cudaMalloc(&d_lut5_i, ninner*sizeof(int));
  cudaMalloc(&d_lut5_j, ninner*sizeof(int));
  cudaMalloc(&d_lut5_k, ninner*sizeof(int));
  cudaMalloc(&d_lut5_l, ninner*sizeof(int));

*/
  cudaMalloc(&d_fivepcf, size_5); 
  cudaMalloc(&d_weight5pcf, size_w); 


  cudaMalloc(&d_alm, nb*nlm*sizeof(thrust::complex<double>));
  cudaMalloc(&d_almconj, nb*nlm*sizeof(thrust::complex<double>));
/*
  double *d_alm_real, *d_alm_imag;
  cudaMallocManaged(&d_alm_real, nb*nlm*sizeof(double));
  cudaMallocManaged(&d_alm_imag, nb*nlm*sizeof(double));
  for (int i = 0; i < nb*nlm; i++) {
    d_alm_real[i] = alm[i].real();
    d_alm_imag[i] = alm[i].imag();
  }
*/

/*
  cudaMemcpy(d_lut5_l1, lut5_l1, size_outer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_l2, lut5_l2, size_outer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_l12, lut5_l12, size_outer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_l3, lut5_l3, size_outer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_l4, lut5_l4, size_outer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_n, lut5_n, size_outer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_zeta, lut5_zeta, size_outer, cudaMemcpyHostToDevice);

  cudaMemcpy(d_lut5_i, lut5_i, size_inner, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_j, lut5_j, size_inner, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_k, lut5_k, size_inner, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lut5_l, lut5_l, size_inner, cudaMemcpyHostToDevice);
*/

  cudaMemcpy(d_alm, alm, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_almconj, almconj, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

  cudaMemcpy(d_fivepcf, fivepcf, size_5, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight5pcf, weight5pcf, size_w, cudaMemcpyHostToDevice);

  if (count % 100 == 0) {
    std::cout << "COUNT " << count << " NOUTER = " << nouter << std::endl;
  } 
  count++;
  int nl = 1;

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter*nl;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2,
        lut5_l12, lut5_l3, lut5_l4, 
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nb, norder, nlm, nouter, ninner);

/*
  test2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf, d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2,
        lut5_l12, lut5_l3, lut5_l4, 
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l, wp, nlm, nouter, ninner, nl);

  test3_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf, d_weight5pcf, d_alm_real, d_alm_imag, lut5_l1, lut5_l2,
        lut5_l12, lut5_l3, lut5_l4, 
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l, wp, nlm, nouter, ninner, nl);
*/

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cudaMemcpy(fivepcf, d_fivepcf, size_5, cudaMemcpyDeviceToHost);
/*
  cudaFree(d_lut5_l1);
  cudaFree(d_lut5_l2);
  cudaFree(d_lut5_l12);
  cudaFree(d_lut5_l3);
  cudaFree(d_lut5_l4);
  cudaFree(d_lut5_n);
  cudaFree(d_lut5_zeta);
  cudaFree(d_lut5_i);
  cudaFree(d_lut5_j);
  cudaFree(d_lut5_k);
  cudaFree(d_lut5_l);
*/
  cudaFree(d_alm);
  cudaFree(d_almconj);
  cudaFree(d_fivepcf);
  cudaFree(d_weight5pcf);
}


void gpu_add_to_power5_orig(double *fivepcf, double *weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  double* d_fivepcf, *d_weight5pcf;
  thrust::complex<double>* d_alm, *d_almconj;
  //size_t size_outer = nouter * sizeof(int);
  //size_t size_inner = ninner * sizeof(int);
  size_t size_w = sizeof(double)*(norder+1)*(norder+1)*(norder+1)*(norder+1)*(2*norder+1)*(norder+1)*(norder+1)*(norder+1);
  size_t size_5 = sizeof(double)*nell5*ninner;

  cudaMalloc(&d_fivepcf, size_5); 
  cudaMalloc(&d_weight5pcf, size_w); 
  cudaMalloc(&d_alm, nb*nlm*sizeof(thrust::complex<double>));
  cudaMalloc(&d_almconj, nb*nlm*sizeof(thrust::complex<double>));

  cudaMemcpy(d_fivepcf, fivepcf, size_5, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight5pcf, weight5pcf, size_w, cudaMemcpyHostToDevice);
  cudaMemcpy(d_alm, alm, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_almconj, almconj, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

  if (count % 100 == 0) {
    std::cout << "OCOUNT " << count << " NOUTER = " << nouter << std::endl;
  } 
  count++;

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel_orig<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cudaMemcpy(fivepcf, d_fivepcf, size_5, cudaMemcpyDeviceToHost);
  cudaFree(d_alm);
  cudaFree(d_almconj);
  cudaFree(d_fivepcf);
  cudaFree(d_weight5pcf);
}

