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
thrust::complex<double>* d_alm, *d_almconj; //define d_alm and d_almconj here
thrust::complex<float>* f_alm, *f_almconj; //for use in float kernels

//* ==== ADD TO POWER 4 KERNELS ==== *//

__global__ void add_to_power4_kernel(double *fourpcf, double *weight4pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_n,
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
        delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real();
	//add to this element
	pcf_element += delta;
      }
    }
    fourpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power4_kernel_float(float *fourpcf, float *weight4pcf,
        thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_n,
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
        delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real();
        //add to this element
        pcf_element += delta;
      }
    }
    fourpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power4_kernel_mixed(double *fourpcf, double *weight4pcf,
        thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_n,
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
        delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real();
        //add to this element
        pcf_element += delta;
      }
    }
    fourpcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power4_kernel_orig(double *fourpcf, double *weight4pcf,
        thrust::complex<double>* alm, thrust::complex<double> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_m1, int *lut4_m2,
	int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nlm, int nouter, int ninner, int almidx) {
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
    double delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real();
    atomicAdd(&fourpcf[bin_index], delta);
}

__global__ void add_to_power4_kernel_orig_float(float *fourpcf,
	float *weight4pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut4_l1, int *lut4_l2,
	int *lut4_l3, int *lut4_m1, int *lut4_m2, int *lut4_n, int *lut4_zeta,
	int *lut4_i, int *lut4_j, int *lut4_k, float wp, 
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
    float delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real();
    atomicAdd(&fourpcf[bin_index], delta);
}

__global__ void add_to_power4_kernel_orig_mixed(double *fourpcf,
	double *weight4pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut4_l1, int *lut4_l2,
	int *lut4_l3, int *lut4_m1, int *lut4_m2, int *lut4_n, int *lut4_zeta,
	int *lut4_i, int *lut4_j, int *lut4_k, float wp, 
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
    double delta = weight*(alm2*alm[almidx+k*nlm+tmp_lm3]).real();
    atomicAdd(&fourpcf[bin_index], delta);
}

//* ==== ADD TO POWER 5 KERNELS ==== *//

__global__ void add_to_power5_kernel(double *fivepcf, double *weight5pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3, int *lut5_l4,
	int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k,
	int *lut5_l, double wp, int nlm, int nouter, int ninner, int almidx) {
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
          delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real();
	  //add to this element
	  pcf_element += delta;
        }
      }
    }
    fivepcf[bin_index] = pcf_element; //copy back to global memory 
}

__global__ void add_to_power5_kernel_float(float *fivepcf, float *weight5pcf,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3, int *lut5_l4,
	int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k,
	int *lut5_l, float wp, int nlm, int nouter, int ninner, int almidx) {
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
          delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real();
          //add to this element
          pcf_element += delta;
        }
      }
    }
    fivepcf[bin_index] = pcf_element; 
}

__global__ void add_to_power5_kernel_mixed(double *fivepcf, double *weight5pcf,
	thrust::complex<float>* alm, thrust::complex<float> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3, int *lut5_l4,
	int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k,
	int *lut5_l, float wp, int nlm, int nouter, int ninner, int almidx) {
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
          delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real();
          //add to this element
          pcf_element += delta;
        }
      }
    }
    fivepcf[bin_index] = pcf_element; 
}

__global__ void add_to_power5_kernel_orig(double *fivepcf, double *weight5pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut5_l1, int *lut5_l2, int *lut5_l3, int *lut5_l4, int *lut5_m1,
	int *lut5_m2, int *lut5_m3, int *lut5_n, int *lut5_zeta,
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
    double delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real();
    atomicAdd(&fivepcf[bin_index], delta);
    //atomicAdd(&fivepcf[bin_index], m2);
}

__global__ void add_to_power5_kernel_orig_float(float *fivepcf,
	float *weight5pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2,
	int *lut5_l3, int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3,
	int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k,
	int *lut5_l, float wp, int nlm, int nouter, int ninner, int almidx) {
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
    float delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real();
    atomicAdd(&fivepcf[bin_index], delta);
}

__global__ void add_to_power5_kernel_orig_mixed(double *fivepcf,
	double *weight5pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2,
	int *lut5_l3, int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3,
	int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k,
	int *lut5_l, float wp, int nlm, int nouter, int ninner, int almidx) {
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
    thrust::complex<float> alm1w = 0;
    thrust::complex<float> alm2 = 0;
    thrust::complex<float> alm3 = 0;
    if (m1 < 0) alm1w = wp*almconj[almidx+ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[almidx+ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[almidx+j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[almidx+j*nlm+tmp_l2+m2];
    int m4 = -m1-m2-m3;
    int tmp_lm4 = tmp_l4+m4;
    if (m3 < 0) alm3 = alm2*almconj[almidx+k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[almidx+k*nlm+tmp_l3+m3];
    double delta = weight*(alm3*alm[almidx+l*nlm+tmp_lm4]).real();
    atomicAdd(&fivepcf[bin_index], delta);
}

//* ==== ADD PAIRS AND MULTIPOLES ==== *//

/****   Add particles methods ****/

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

    for (int k = st; k < st+np; k++) {
      if (samecell && j == k) continue;

      dx = posx[k] - posx[j];
      dy = posy[k] - posy[j];
      dz = posz[k] - posz[j];
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
      pair_w = w[k]*w[j];
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
      pair_w = w[k]*w[j];
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


//* ==== CPU METHODS ==== *//
//* ==== Allocate LUTs 4 ==== *//

void gpu_allocate_luts4(int **p_lut4_l1, int **p_lut4_l2, int **p_lut4_l3, int **p_lut4_n,
	int **p_lut4_zeta, int **p_lut4_i, int **p_lut4_j, int **p_lut4_k,
        int nouter, int ninner) {
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&(*p_lut4_l1), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut4_l2), nouter*sizeof(int));
  cudaMallocManaged(&(*p_lut4_l3), nouter*sizeof(int));
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

void gpu_free_luts4(int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_n,
	int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k) {
  cudaFree(lut4_l1);
  cudaFree(lut4_l2);
  cudaFree(lut4_l3);
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

//* ==== Allocate LUTs 5 ==== *//

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
        int *lut5_l4, int *lut5_n, int *lut5_zeta, int *lut5_i,
	int *lut5_j, int *lut5_k, int *lut5_l) {
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

//* ==== ALLOCATE MULTIPOLES AND PARTICLES ==== *//

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

//* ==== ADD TO POWER 4 METHODS ===== *//

void gpu_add_to_power4(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_n, int *lut4_zeta,
	int *lut4_i, int *lut4_j, int *lut4_k, double wp, int nb,
	int nlm, int nouter, int ninner, int nell4) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

if (count == 0) {
count++;
std::cout << "Threads = " << threads << " Nouter = " << nouter << " Ninner = " << ninner << std::endl;
}

  add_to_power4_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2,
        lut4_l3, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k, 
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//float version of main kernel
void gpu_add_to_power4_float(float *d_fourpcf, float *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_float<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2,
        lut4_l3, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//mixed precision
void gpu_add_to_power4_mixed(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2,
        lut4_l3, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//alternate (original) kernel
void gpu_add_to_power4_orig(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nb, int nlm, int nouter, int ninner,
	int nell4) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

if (count == 0) {
count++;
std::cout << "Threads = " << threads << std::endl;
}

  add_to_power4_kernel_orig<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//float version
void gpu_add_to_power4_orig_float(float *d_fourpcf, float *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_orig_float<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2, lut4_l3,
        lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//mixed precision
void gpu_add_to_power4_orig_mixed(double *d_fourpcf, double *d_weight4pcf, 
        int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int nlm, int nouter, int ninner, int nell4) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart*nb*nlm;
  pstart++;

  add_to_power4_kernel_orig_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, f_alm, f_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

//* ==== ADD TO POWER 5 METHODS ===== *//

void gpu_add_to_power5(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2,
        lut5_l12, lut5_l3, lut5_l4, 
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_float(float *d_fivepcf, float *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_float<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
	d_weight5pcf, f_alm, f_almconj, lut5_l1, lut5_l2, lut5_l12,
	lut5_l3, lut5_l4, lut5_n, lut5_zeta, lut5_i,
	lut5_j, lut5_k, lut5_l, wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_mixed(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
	d_weight5pcf, f_alm, f_almconj, lut5_l1, lut5_l2, lut5_l12,
	lut5_l3, lut5_l4, lut5_n, lut5_zeta, lut5_i, lut5_j,
	lut5_k, lut5_l, wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_orig(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //d_alm and d_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_orig<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_orig_float(float *d_fivepcf, float *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_orig_float<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, f_alm, f_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_to_power5_orig_mixed(double *d_fivepcf, double *d_weight5pcf, 
        int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int nlm, int nouter, int ninner, int nell5) {

  //f_alm and f_almconj already allocated and computed

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  //calculate index of d_alm for this particle
  int almidx = pstart5*nb*nlm;
  pstart5++;

  add_to_power5_kernel_orig_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, f_alm, f_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nlm, nouter, ninner, almidx);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
}

void gpu_add_pairs_and_multipoles(double *m, double *posx, double *posy,
        double *posz, double *w, int *ct, int *pnum, int *spnum,
	int *snp, int *sc, double *x0i, double *x2i, int n, int nbin,
	int order, int nmult, float rmin, float rmax, int pstart5) {
  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_pairs_and_multipoles_kernel<<<blocksPerGrid, threadsPerBlock>>>(m,
        posx, posy, posz, w, ct, pnum, spnum, snp, sc, x0i, x2i,
        n, nbin, order, nmult, rmin, rmax, rmin2, rmax2, pstart5);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

void gpu_add_pairs_and_multipoles_periodic(double *m, double *posx,
        double *posy, double *posz, double *w, int *ct, int *pnum, int *spnum,
        int *snp, int *sc, double *x0i, double *x2i, int *delta_x,
        int *delta_y, int *delta_z, int n, int nbin, int order, int nmult,
        float rmin, float rmax, int pstart5, double cellsize) {
  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = n;
  float rmin2 = rmin*rmin;
  float rmax2 = rmax*rmax;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_pairs_and_multipoles_periodic_kernel<<<blocksPerGrid,
	threadsPerBlock>>>(m,
        posx, posy, posz, w, ct, pnum, spnum, snp, sc, x0i, x2i,
        delta_x, delta_y, delta_z, n, nbin, order, nmult, rmin, rmax, rmin2,
	rmax2, pstart5, cellsize);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

void gpu_compute_alms(int *map, double *m, int nbin, int nlm, int maxp,
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
  int threadsPerBlock = 512;
  long threads = maxp*nbin;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  compute_alms<<<blocksPerGrid, threadsPerBlock>>>(d_alm, d_almconj, d_map, m,
	nbin, nlm, maxp, order, mapdim, nmult);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
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
  int threadsPerBlock = 512;
  long threads = maxp*nbin;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  compute_alms_float<<<blocksPerGrid, threadsPerBlock>>>(f_alm, f_almconj,
	d_map, m, nbin, nlm, maxp, order, mapdim, nmult);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}
