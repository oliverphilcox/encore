#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include "gpufuncs.h"
#include <thrust/complex.h>
#include <cuComplex.h>

int count = 0;

__global__ void add_to_power4_kernel(double *fourpcf, double *weight4pcf,
	thrust::complex<double>* alm, thrust::complex<double> *almconj,
	int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_n,
	int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k, 
        double wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
      if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
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
        if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
        
	//calculate delta
        delta = weight*(alm2*alm[k*nlm+tmp_lm3]).real();
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
        float wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
      if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
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
        if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];

        //calculate delta
        delta = weight*(alm2*alm[k*nlm+tmp_lm3]).real();
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
        float wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
      if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
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
        if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];

        //calculate delta
        delta = weight*(alm2*alm[k*nlm+tmp_lm3]).real();
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
        double wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
    if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
    int m3 = -m1-m2;
    int tmp_lm3 = tmp_l3+m3;
    double delta = weight*(alm2*alm[k*nlm+tmp_lm3]).real();
    atomicAdd(&fourpcf[bin_index], delta);
}

__global__ void add_to_power4_kernel_orig_float(float *fourpcf, float *weight4pcf,
        thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
    if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
    int m3 = -m1-m2;
    int tmp_lm3 = tmp_l3+m3;
    float delta = weight*(alm2*alm[k*nlm+tmp_lm3]).real();
    atomicAdd(&fourpcf[bin_index], delta);
}

__global__ void add_to_power4_kernel_orig_mixed(double *fourpcf, double *weight4pcf,
        thrust::complex<float>* alm, thrust::complex<float> *almconj,
        int *lut4_l1, int *lut4_l2, int *lut4_l3, int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
    if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
    int m3 = -m1-m2;
    int tmp_lm3 = tmp_l3+m3;
    double delta = weight*(alm2*alm[k*nlm+tmp_lm3]).real();
    atomicAdd(&fourpcf[bin_index], delta);
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

__global__ void add_to_power5_kernel_float(float *fivepcf, float *weight5pcf, thrust::complex<float>* alm,
        thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	float wp, int nlm, int nouter, int ninner) {
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
    fivepcf[bin_index] = pcf_element; 
}

__global__ void add_to_power5_kernel_mixed(double *fivepcf, double *weight5pcf, thrust::complex<float>* alm,
        thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n, int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	float wp, int nlm, int nouter, int ninner) {
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
    fivepcf[bin_index] = pcf_element; 
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

__global__ void add_to_power5_kernel_orig_float(float *fivepcf, float *weight5pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n, int *lut5_zeta,
	int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
    if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
    int m4 = -m1-m2-m3;
    int tmp_lm4 = tmp_l4+m4;
    if (m3 < 0) alm3 = alm2*almconj[k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[k*nlm+tmp_l3+m3];
    float delta = weight*(alm3*alm[l*nlm+tmp_lm4]).real();
    atomicAdd(&fivepcf[bin_index], delta);
}

__global__ void add_to_power5_kernel_orig_mixed(double *fivepcf, double *weight5pcf, thrust::complex<float>* alm,
	thrust::complex<float> *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n, int *lut5_zeta,
	int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner) {
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
    if (m1 < 0) alm1w = wp*almconj[ii*nlm+tmp_l1-m1]; else alm1w = wp*alm[ii*nlm+tmp_l1+m1];
    if (m2 < 0) alm2 = alm1w*almconj[j*nlm+tmp_l2-m2]; else alm2 = alm1w*alm[j*nlm+tmp_l2+m2];
    int m4 = -m1-m2-m3;
    int tmp_lm4 = tmp_l4+m4;
    if (m3 < 0) alm3 = alm2*almconj[k*nlm+tmp_l3-m3]; else alm3 = alm2*alm[k*nlm+tmp_l3+m3];
    double delta = weight*(alm3*alm[l*nlm+tmp_lm4]).real();
    atomicAdd(&fivepcf[bin_index], delta);
}

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

void gpu_add_to_power4(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4) {

  thrust::complex<double>* d_alm, *d_almconj;
  cudaMalloc(&d_alm, nb*nlm*sizeof(thrust::complex<double>));
  cudaMalloc(&d_almconj, nb*nlm*sizeof(thrust::complex<double>));

  cudaMemcpy(d_alm, alm, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_almconj, almconj, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

if (count == 0) {
count++;
std::cout << "Threads = " << threads << " Nouter = " << nouter << " Ninner = " << ninner << std::endl;
}

  add_to_power4_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2,
        lut4_l3, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k, 
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

//float version of main kernel
void gpu_add_to_power4_float(float *d_fourpcf, float *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4) {

  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power4_kernel_float<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2,
        lut4_l3, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

//mixed precision
void gpu_add_to_power4_mixed(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4) {

  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power4_kernel_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2,
        lut4_l3, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

//alternate (original) kernel
void gpu_add_to_power4_orig(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4) {
  thrust::complex<double>* d_alm, *d_almconj;
  cudaMalloc(&d_alm, nb*nlm*sizeof(thrust::complex<double>));
  cudaMalloc(&d_almconj, nb*nlm*sizeof(thrust::complex<double>));

  cudaMemcpy(d_alm, alm, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_almconj, almconj, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

if (count == 0) {
count++;
std::cout << "Threads = " << threads << std::endl;
}

  add_to_power4_kernel_orig<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

//float version
void gpu_add_to_power4_orig_float(float *d_fourpcf, float *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4) {
  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power4_kernel_orig_float<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

//mixed precision
void gpu_add_to_power4_orig_mixed(double *d_fourpcf, double *d_weight4pcf, Complex* alm,
        Complex *almconj, int *lut4_l1, int *lut4_l2, int *lut4_l3,
        int *lut4_m1, int *lut4_m2,
        int *lut4_n, int *lut4_zeta, int *lut4_i, int *lut4_j, int *lut4_k,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell4) {
  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power4_kernel_orig_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fourpcf,
        d_weight4pcf, d_alm, d_almconj, lut4_l1, lut4_l2, lut4_l3,
	lut4_m1, lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

void gpu_add_to_power5(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  thrust::complex<double>* d_alm, *d_almconj;
  cudaMalloc(&d_alm, nb*nlm*sizeof(thrust::complex<double>));
  cudaMalloc(&d_almconj, nb*nlm*sizeof(thrust::complex<double>));

  cudaMemcpy(d_alm, alm, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_almconj, almconj, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2,
        lut5_l12, lut5_l3, lut5_l4, 
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

void gpu_add_to_power5_float(float *d_fivepcf, float *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel_float<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
	d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2, lut5_l12,
	lut5_l3, lut5_l4, lut5_n, lut5_zeta, lut5_i,
	lut5_j, lut5_k, lut5_l, wp, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

void gpu_add_to_power5_mixed(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
	d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2, lut5_l12,
	lut5_l3, lut5_l4, lut5_n, lut5_zeta, lut5_i, lut5_j,
	lut5_k, lut5_l, wp, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaFree(d_alm);
  cudaFree(d_almconj);
}

void gpu_add_to_power5_with_memcpy(double *fivepcf, double *weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
        int *lut5_l4, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {


  double* d_fivepcf, *d_weight5pcf;
  thrust::complex<double>* d_alm, *d_almconj;
  size_t size_w = sizeof(double)*(norder+1)*(norder+1)*(norder+1)*(norder+1)*(2*norder+1)*(norder+1)*(norder+1)*(norder+1);
  size_t size_5 = sizeof(double)*nell5*ninner;

  cudaMalloc(&d_fivepcf, size_5); 
  cudaMalloc(&d_weight5pcf, size_w); 
  cudaMalloc(&d_alm, nb*nlm*sizeof(thrust::complex<double>));
  cudaMalloc(&d_almconj, nb*nlm*sizeof(thrust::complex<double>));

  cudaMemcpy(d_alm, alm, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_almconj, almconj, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fivepcf, fivepcf, size_5, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight5pcf, weight5pcf, size_w, cudaMemcpyHostToDevice);

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1, lut5_l2,
        lut5_l12, lut5_l3, lut5_l4, 
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  //Copy memory back to host in this method
  cudaMemcpy(fivepcf, d_fivepcf, size_5, cudaMemcpyDeviceToHost);
  //have to free memory
  cudaFree(d_alm);
  cudaFree(d_almconj);
  cudaFree(d_fivepcf);
  cudaFree(d_weight5pcf);
}

void gpu_add_to_power5_orig(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  thrust::complex<double>* d_alm, *d_almconj;

  cudaMalloc(&d_alm, nb*nlm*sizeof(thrust::complex<double>));
  cudaMalloc(&d_almconj, nb*nlm*sizeof(thrust::complex<double>));

  cudaMemcpy(d_alm, alm, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_almconj, almconj, nb*nlm*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

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

  cudaFree(d_alm);
  cudaFree(d_almconj);
}

void gpu_add_to_power5_orig_float(float *d_fivepcf, float *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel_orig_float<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cudaFree(d_alm);
  cudaFree(d_almconj);
}

void gpu_add_to_power5_orig_mixed(double *d_fivepcf, double *d_weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
        float wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  thrust::complex<float>* d_alm, *d_almconj;
  cudaMallocManaged(&d_alm, nb*nlm*sizeof(thrust::complex<float>));
  cudaMallocManaged(&d_almconj, nb*nlm*sizeof(thrust::complex<float>));

  for (int i = 0; i < nb*nlm; i++) {
    d_alm[i] = (thrust::complex<float>)(alm[i]);
    d_almconj[i] = (thrust::complex<float>)(almconj[i]);
  }

  // Invoke kernel
  int threadsPerBlock = 512;
  long threads = ninner*nouter;
  int blocksPerGrid = (threads+threadsPerBlock-1) / threadsPerBlock;

  add_to_power5_kernel_orig_mixed<<<blocksPerGrid, threadsPerBlock>>>(d_fivepcf,
        d_weight5pcf, d_alm, d_almconj, lut5_l1,lut5_l2,
        lut5_l3, lut5_l4, lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, nb, norder, nlm, nouter, ninner);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cudaFree(d_alm);
  cudaFree(d_almconj);
}

void gpu_add_to_power5_orig_with_memcpy(double *fivepcf, double *weight5pcf, Complex* alm,
        Complex *almconj, int *lut5_l1, int *lut5_l2, int *lut5_l3,
        int *lut5_l4, int *lut5_m1, int *lut5_m2, int *lut5_m3, int *lut5_n,
        int *lut5_zeta, int *lut5_i, int *lut5_j, int *lut5_k, int *lut5_l,
	double wp, int nb, int norder, int nlm, int nouter, int ninner, int nell5) {

  double* d_fivepcf, *d_weight5pcf;
  thrust::complex<double>* d_alm, *d_almconj;
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
