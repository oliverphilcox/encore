#ifndef NPCF_H
#define NPCF_H

// ========================== Here are all of the cross-powers ============

#include "WeightFunctions.h"

class NPCF {
// This should accumulate the NPCF contributions, for all combination of bins.
  public:
    uint64 bincounts[NBIN];
    Float binweight[NBIN];
    int map[MAXORDER+1][MAXORDER+1][MAXORDER+1];   // The multipole index of x^a y^b z^c
    STimer MultTimer;
    STimer BinTimer3;

    // Array to hold the 3PCF
    #define NL (ORDER+1)
    #define N3PCF (NBIN*(1+NBIN)/2)
    Float threepcf[N3PCF*NL];

#ifdef FOURPCF
    STimer BinTimer4;

// Sizes of 4pcf array
#define N4PCF (NBIN*(1+NBIN)*(2+NBIN)/6)

    // Array to hold the 4PCF (some bins will violate triangle condition / parity and be empty
    Float *fourpcf;
    // length of angular part of 4PCF
    int nell4;
#endif


    void make_map() {
	// Construct the index number in our multipoles for x^a y^b z^c
        for(int i=0;i<=MAXORDER;i++)
            for(int j=0;j<=MAXORDER-i;j++)
                for(int k=0;k<=MAXORDER-i-j;k++) map[i][j][k] = 0;
	int n=0;
        for(int i=0;i<=ORDER;i++)
            for(int j=0;j<=ORDER-i;j++)
                for(int k=0;k<=ORDER-i-j;k++) {
		    map[i][j][k] = n; n++;
		}
	return;
    }

    void reset() {
	// Zero out the array on construction.
	for (int i=0, ct=0; i<NBIN; i++) {
	    bincounts[i] = 0;
	    binweight[i] = 0.0;
	    for (int j=i; j<NBIN; j++, ct++) {
    		for (int l_i=0; l_i<=ORDER; l_i++) {
    		    threepcf[l_i*N3PCF+ct] = 0.0;
    		}
      }
    }
  #ifdef FOURPCF

    // First work out how long the 4PCF array should be
    nell4=0;
    for(int l1=0;l1<=ORDER;l1++){
      for(int l2=0;l2<=ORDER;l2++){
        for(int l3=fmax(0,fabs(l1-l2));l3<=fmin(ORDER,l1+l2);l3++) nell4++;
      }
    }
    // Now allocate memory
    fourpcf = (Float *)malloc(sizeof(Float)*nell4*N4PCF);

    // Initialize array to zero
    for(int l=0;l<nell4;l++){
      for(int i=0; i<N4PCF;i++) fourpcf[l*N4PCF+i] = 0.0;
    }
  #endif
	    return;
    }

    NPCF() {
	make_map();
        reset();
	return;
    }
    ~NPCF() {
    }

    void sum_power(NPCF *c) {
	// Just add up all of the threaded power into the zeroth element

  	for (int i=0; i<NBIN; i++) {
  	    bincounts[i] += c->bincounts[i];
  	    binweight[i] += c->binweight[i];
  }

  for(int x=0;x<N3PCF*NL;x++) threepcf[x] += c->threepcf[x];

  #ifdef FOURPCF
  for(int x=0;x<N4PCF*nell4;x++) fourpcf[x] += c->fourpcf[x];
  #endif
    }

    void report_power() {
      /// Report the NPCF measurements.

      // NB: we print zeta_ij[ell] for all j<=i
      // Old versions printed zeta_ij[ell]/zeta_ij[0] and in a different order
	for (int i=0, ct=0; i<NBIN; i++) {
	    for (int j=i; j<NBIN; j++,ct++) {
		if (j==i) printf("Multipole Power %2d %9lld %9.0f\n",
				j, bincounts[j], binweight[j]);
		//if (j==i) {
		//    printf("# Bin auto-power omitted due to uncorrected noise bias\n");
		//    continue;
		//}
		printf("%2d %2d", i, j);
		for (int l=0; l<=ORDER; l++) {
		    printf(" %13.6e", threepcf[l*N3PCF+ct]); ///threepcf[i][j][0]);
		}
	    printf("\n");
	    }
	}
  #ifdef FOURPCF
  printf("##TODO: Add 4PCF Here\n");
  #endif
    }

    void report_timings() {
      /// Report the NPCF timing measurements (for a single CPU).

      printf("\n# Single CPU Timings");
      printf("\nSpherical harmonics: %.3f s",MultTimer.Elapsed());
      printf("\n3PCF binning: %.3f s",BinTimer3.Elapsed());
      #ifdef FOURPCF
        printf("\n4PCF binning: %.3f s",BinTimer4.Elapsed());
      #endif
      printf("\n");
    }

    void save_power(char* out_string, Float rmax) {
      // Print the output NPCF counts to file

      // SAVE 3PCF

      // First create output files
       char out_name[1000];
        snprintf(out_name, sizeof out_name, "output/%s_3pcf.txt", out_string);
       FILE * OutFile = fopen(out_name,"w");

       // Print some useful information
       fprintf(OutFile,"## Order: %d\n",ORDER);
       fprintf(OutFile,"## Bins: %d\n",NBIN);
       fprintf(OutFile,"## Maximum Radius = %.2e\n", rmax);
       fprintf(OutFile,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Rows 3+ = zeta_ell^ab\n");
       fprintf(OutFile,"## Column 1 specifies the angular multipole\n");

       // First print the indices of the first radial bin
       fprintf(OutFile," \t"); // empty l1 specifier
       for(int i=0;i<NBIN;i++){
         for(int j=i; j<NBIN; j++) fprintf(OutFile,"%2d\t",i);
       }
       fprintf(OutFile," \n");

       // Print the indices of the second radial bin
       fprintf(OutFile,"\t"); // empty ell specifier
       for(int i=0;i<NBIN;i++){
         for(int j=i; j<NBIN; j++) fprintf(OutFile,"%2d\t",j);
       }
       fprintf(OutFile,"\n");

       // Now print the 3PCF, ell-by-ell.
       for(int ell=0;ell<=ORDER;ell++){
         fprintf(OutFile,"%d\t",ell);

           for (int i=0, ct=0; i<NBIN; i++){
             for(int j=i; j<NBIN; j++, ct++){
                fprintf(OutFile,"%le\t",threepcf[ell*N3PCF+ct]);
              }
          }
          fprintf(OutFile,"\n");
        }

       fflush(NULL);

       // Close open files
       fclose(OutFile);

       printf("\n3PCF Output saved to %s\n",out_name);

       #ifdef FOURPCF

       // SAVE 4PCF

       // First create output files
       char out_name2[1000];
       snprintf(out_name2, sizeof out_name2, "output/%s_4pcf.txt", out_string);
       FILE * OutFile2 = fopen(out_name2,"w");

        // Print some useful information
        fprintf(OutFile2,"## Order: %d\n",ORDER);
        fprintf(OutFile2,"## Bins: %d\n",NBIN);
        fprintf(OutFile2,"## Maximum Radius = %.2e\n", rmax);
        fprintf(OutFile2,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Rows 4+ = zeta_l1l2l3^abc\n");
        fprintf(OutFile2,"## Columns 1-3 specify the (l1, l2, l3) multipole triplet\n");

        // First print the indices of the radial bins
        fprintf(OutFile2,"\t\t\t"); // empty l1,l2,l3 specifier
        for(int i=0;i<NBIN;i++){
          for(int j=i; j<NBIN; j++){
            for(int k=j; k<NBIN; k++){
              fprintf(OutFile2,"%2d\t",i);
            }
          }
        }
        fprintf(OutFile2,"\n");

        fprintf(OutFile2,"\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i; j<NBIN; j++){
            for(int k=j; k<NBIN; k++){
              fprintf(OutFile2,"%2d\t",j);
            }
          }
        }
        fprintf(OutFile2,"\n");

        fprintf(OutFile2,"\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i; j<NBIN; j++){
            for(int k=j; k<NBIN; k++){
              fprintf(OutFile2,"%2d\t",k);
            }
          }
        }
        fprintf(OutFile2,"\n");

        // Now print the 4PCF, ell-by-ell.
        for(int l1=0, l_index=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
            for(int l3=fmax(0,fabs(l1-l2));l3<=fmin(ORDER,l1+l2);l3++,l_index++){
              if(pow(-1.,l1+l2+l3)==-1) continue; // skip odd parity and triangle violating bins
              fprintf(OutFile2,"%d\t%d\t%d\t",l1,l2,l3);
              for (int i=0;i<N4PCF;i++) fprintf(OutFile2,"%le\t",fourpcf[N4PCF*l_index+i]);
             fprintf(OutFile2,"\n");
            }
          }
        }
        fflush(NULL);

        // Close open files
        fclose(OutFile2);

        printf("\n4PCF Output saved to %s\n",out_name2);


       #endif

     }


    inline void add_to_power(Multipoles *mult, Float wp) {
      // wp is the primary galaxy weight
	// Now use all of the binned multipoles to compute the
	// spherical harmonics in all bins and then the cross-powers.
	// Need some scratch space:
	// This also applies the weight of the primary galaxy.
	Complex alm[NBIN][NLM];   // Apparently this initializes to zero

  MultTimer.Start();

	for (int i=0; i<NBIN; i++) {
	    Float *m = mult[i].multipoles();
        bincounts[i] += mult[i].ncount();
	    binweight[i] += m[0];

	    // Now we're going to recast the Cartesian multipoles into the Y_lm's.
	    // IMPORTANT: Because we used unit vectors, we get to use
	    // cos(theta) = z   and sin(theta) exp(i*phi) = (x+iy)
	    // For example, we get to say
	    // Y_30 propto 5*cos^3 - 3*cos = 5z^3 - 3z
	    // instead of having to use
	    // 5z^3 - 3zr^2 = 2z^3 - 3zx^2 - 3zy^2
	    // This reduces the number of terms at high ell.
	    // But it means that not all terms have the same sum of exponents!

// Bring in all of the tedious spherical harmonic code from another file.
#define CM(a,b,c) m[map[a][b][c]]
	    Complex *almbin = &(alm[i][0]);
#include "spherical_harmonics.cpp"

	 }

   MultTimer.Stop();

#define RealProduct(a,b) (a.real()*b.real()+a.imag()*b.imag())

  // COMPUTE 3PCF CONTRIBUTIONS

  BinTimer3.Start();


	for (int i=0, ct=0; i<NBIN; i++) {
	    for (int j=i; j<NBIN; j++, ct++) {
  		  // Fill in a triangle of threepcf radial bins.
    		// We want to compute the sum over m of Ylm[i] Yl(-m)[j] = Ylm[i]Ylm*[j](-1)^m
    		// For m=0, that's just a_l0^2
    		// For others, (a+bi)*(A-Bi) + (a-bi)*(A+Bi) = 2*(a*A+b*B)
    		// Note we use a different normalization convention to that of Slepian/Eisenstein 2015

        // Effectively we recast Sum_{m} a_m b_{-m} to Sum_{m>=0} (-1)^m * sym * F[a_m, b_m]
        // where sym = 1 if m=0 and 2 else, and F[a, b] = Re[a]Re[b]+Im[a]Im[b]
        // We've put that factor of 2 and the (-1)^m in the weight3pcf[] array.

    		for (int ell=0, n=0; ell<=ORDER; ell++) {
  		    for (int mm=0; mm<=ell; mm++, n++) {
      			// n counts the (ell,m)
            // indexing isn't super efficient here, but matches that of 4PCF (where it is more important)
      			// Our definition is that the power is (-1)^ell / Sqrt(2ell+1) * Sum_m a_lm[a] * a_lm[b].conj()
      			threepcf[ell*N3PCF+ct] +=
      			    wp*RealProduct(alm[i][n],alm[j][n])*weight3pcf[n]; //* 4.0 * M_PI / (2*ell+1.0);
          }
        }
      }
    }

  BinTimer3.Stop();

#ifdef FOURPCF
  // COMPUTE 4PCF CONTRIBUTIONS

  // Define Real[a b c^*]
  //TODO: test if faster if we just take the real part of the product...
  #define RealProduct3(a,b,c) ((a*b*c).real())

  BinTimer4.Start();
  //
  // Float weight1, weight2; // coupling weights
  // int tmp_lm1, tmp_lm2, tmp_lm3; // indices
  // Complex alm3;
  //
  // // Compute unravelled arrays for alm and conjugate
  // Complex alm_ravel[NBIN*NLM], almconj_ravel[NBIN*NLM];
  // for(int x=0, ct=0; x<NLM;x++){
  //   for(int y=0; y<NBIN;y++, ct++){
  //     alm_ravel[ct] = alm[y][x];
  //     almconj_ravel[ct] = conj(alm[y][x]);
  //   }
  // }
  //
  // // Now iterate over (l1, l2, l3) triplet.
  // // NB: n indexes position in the 4PCF weight array
  // // We only compute terms with even parity i.e. l1+l2+l3. These are all real.
  // // The odd parity terms could be included if necessary and are purely imaginary
  //
  // // Iterate over first multipole
  // for(int l1=0, zeta_index=0, n=0; l1<=ORDER; l1++) {
  //
  //     // Iterate over second multipole
  //     for(int l2=0; l2<=ORDER; l2++){
  //
  //       // Iterate over third multipole, avoiding bins violating triangle condition
  //       for(int l3=fmax(0,fabs(l1-l2));l3<=fmin(ORDER,l1+l2);l3++,zeta_index+=N4PCF){
  //
  //         // Skip any odd multipoles with odd parity
  //         if(pow(-1,l1+l2+l3)==-1) continue;
  //
  //         // Iterate over m1, starting at zero
  //         for(int m1=0; m1<=l1; m1++){
  //
  //           tmp_lm1 = (l1*(l1+1)/2+m1)*NBIN;
  //
  //           // Iterate over m2, starting at zero
  //           for(int m2=0; m2<=l2; m2++, n++){
  //
  //             // Store relevant weights
  //             weight1 = weight4pcf1[n];
  //             weight2 = weight4pcf2[n]; // this will be zero if triangle / summation conditions are violated
  //
  //             // skip trivial case!
  //             if((weight1==0)&&(weight2==0)) continue;
  //
  //             tmp_lm2 = (l2*(l2+1)/2+m2)*NBIN;
  //             tmp_lm3 = (l3*(l3+1)/2)*NBIN;
  //
  //             // Now fill up the 4PCF. We have two sets of (l1,l2,l3,m1,m2,m3) to do.
  //             // It's fastest to check if the weights are non-zero then do one or both of these
  //             // Lossy to check non-zero weight in every bin!
  //
  //             // Use first set only
  //             if((weight1!=0)&&(weight2==0)){
  //
  //               // Iterate over first radial bin in lower hypertriangle
  //               for(int i=0, bin_index=zeta_index; i<=NBIN; i++){
  //
  //                 // Iterate over second bin
  //                 for(int j=0; j<=i; j++){
  //                   // Iterate over final bin
  //                   for(int k=0; k<=j; k++, bin_index++){
  //
  //                     if(m2>=m1) alm3 = wp*almconj_ravel[tmp_lm3+(m2-m1)*NBIN+k]*weight1;
  //                     else alm3 = wp*alm_ravel[tmp_lm3*(m1-m2)+NBIN+k]*weight1*pow(-1.,m1-m2); // take conjugate if necessary
  //
  //                     fourpcf[bin_index] += (alm_ravel[tmp_lm1+i]*alm_ravel[tmp_lm2+j]*alm3).real();
  //                   }
  //                 }
  //               }
  //             }
  //
  //             // Use second set only
  //             if((weight1==0)&&(weight2!=0)){
  //
  //               // Iterate over first radial bin in lower hypertriangle
  //               for(int i=0, bin_index=zeta_index; i<=NBIN; i++){
  //
  //                 // Iterate over second bin
  //                 for(int j=0; j<=i; j++){
  //
  //                   // Iterate over final bin
  //                   for(int k=0; k<=j; k++, bin_index++){
  //
  //                     if(m2>=m1) alm3 = wp*alm_ravel[tmp_lm3+(m2-m1)*NBIN+k]*weight2;
  //                     else alm3 = wp*almconj_ravel[tmp_lm3+(m1-m2)*NBIN+k]*weight2*pow(-1.,m1-m2);
  //
  //                     fourpcf[bin_index] += (alm_ravel[tmp_lm1+i]*almconj_ravel[tmp_lm2+j]*alm3).real();
  //                   }
  //                 }
  //               }
  //             }
  //
  //             // Use both sets
  //             if((weight1!=0)&&(weight2!=0)){
  //               for(int i=0, bin_index=zeta_index; i<=NBIN; i++){
  //
  //                 // Iterate over second bin
  //                 for(int j=0; j<=i; j++){
  //
  //                   // Iterate over final bin
  //                   for(int k=0; k<=j; k++, bin_index++){
  //
  //                     // First combination
  //                     if(m2>=m1) alm3 = wp*almconj_ravel[tmp_lm3+(m2-m1)*NBIN+k]*weight1;
  //                     else alm3 = wp*alm_ravel[tmp_lm3+(m1-m2)*NBIN+k]*weight1*pow(-1.,m1-m2); // take conjugate if necessary
  //
  //                     fourpcf[bin_index] += (alm_ravel[tmp_lm1+i]*alm_ravel[tmp_lm2+j]*alm3).real();
  //
  //                     if(m2>=m1) alm3 = wp*alm_ravel[tmp_lm3+(m2-m1)*NBIN+k]*weight2;
  //                     else alm3 = wp*almconj_ravel[tmp_lm3+(m1-m2)*NBIN+k]*weight2*pow(-1.,m1-m2);
  //
  //                     fourpcf[bin_index] += (alm_ravel[tmp_lm1+i]*almconj_ravel[tmp_lm2+j]*alm3).real();
  //                   }
  //                 }
  //               }
  //             }
  //             //End of radial binning loops
  //           }
  //         }
  //       }
  //     }
  //   }
  //
  //


  Float weight1, weight2; // coupling weights
  int tmp_lm1, tmp_lm2, tmp_lm3;
  Complex alm1w, alm2, alm3, alm2conj, alm3conj; // temporary storage of alm weights
  Complex alm1wlist[NBIN], alm2list[NBIN], alm3list1[NBIN], alm3list2[NBIN]; // arrays to hold intermediate a_lm lists
  Complex alm2conjlist[NBIN], alm3conjlist1[NBIN]; // arrays to hold intermediate a_lm lists

  //NB: do we need all these lists??

  // Precompute complex conjugates of all alm
  Complex almconj[NBIN][NLM];
  for(int x=0;x<NBIN;x++){
    for(int l=0, y=0;l<=ORDER;l++){
      for(int m=0;m<=l;m++,y++) almconj[x][y] = conj(alm[x][y]);
    }
  }

  // Now iterate over (l1, l2, l3) triplet.
  // NB: n indexes position in the 4PCF weight array
  // We only compute terms with even parity i.e. l1+l2+l3. These are all real.
  // The odd parity terms could be included if necessary and are purely imaginary

  // Iterate over first multipole
  for(int l1=0, zeta_index=0, n=0; l1<=ORDER; l1++) {

      // Iterate over second multipole
      for(int l2=0; l2<=ORDER; l2++){

        // Iterate over third multipole, avoiding bins violating triangle condition
        for(int l3=fmax(0,fabs(l1-l2));l3<=fmin(ORDER,l1+l2);l3++,zeta_index+=N4PCF){

          // Skip any odd multipoles with odd parity
          if(pow(-1,l1+l2+l3)==-1) continue;

          // Iterate over m1, starting at zero
          for(int m1=0; m1<=l1; m1++){

            tmp_lm1 = l1*(l1+1)/2+m1;

            // Create temporary copy of a_l1m1
            for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*alm[x][tmp_lm1];

            // Iterate over m2, starting at zero
            for(int m2=0; m2<=l2; m2++, n++){

              // Store relevant weights
              weight1 = weight4pcf1[n];
              weight2 = weight4pcf2[n]; // this will be zero if triangle / summation conditions are violated

              // skip trivial case!
              if((weight1==0)&&(weight2==0)) continue;

              tmp_lm2 = l2*(l2+1)/2+m2;
              tmp_lm3 = l3*(l3+1)/2;

              // Create temporary copy of a_l2m2 and a_l3m3 with correct complex conjugation factors
              // also include the coupling weight
              for(int x=0;x<NBIN;x++){
                alm2list[x] = alm[x][tmp_lm2];
                alm2conjlist[x] = almconj[x][tmp_lm2];
                alm3list1[x] = alm[x][tmp_lm3+m1+m2]*weight1;
                alm3conjlist1[x] = almconj[x][tmp_lm3+m1+m2]*weight1;
                if(m2>=m1) alm3list2[x] = alm[x][tmp_lm3+m2-m1]*weight2;
                else alm3list2[x] = almconj[x][tmp_lm3+m1-m2]*weight2; // take conjugate if necessary (extra factor of (-1)^{m1-m2} is absorbed into the weights)
              }

              // Now fill up the 4PCF. We have two sets of (l1,l2,l3,m1,m2,m3) to do.
              // It's fastest to check if the weights are non-zero then do one or both of these
              // Lossy to check non-zero weight in every bin!

              // Use first set only
              if((weight1!=0)&&(weight2==0)){
                // Iterate over first radial bin in lower hypertriangle
                for(int i=0, bin_index=zeta_index; i<NBIN; i++){

                  alm1w = alm1wlist[i];

                  // Iterate over second bin
                  for(int j=i; j<NBIN; j++){

                    alm2 = alm2list[j];

                    // Iterate over final bin
                    for(int k=j; k<NBIN; k++, bin_index++){

                      fourpcf[bin_index] += (alm1w*alm2*alm3conjlist1[k]).real();
                    }
                  }
                }
              }

              // Use second set only
              if((weight1==0)&&(weight2!=0)){
                // Iterate over first radial bin in lower hypertriangle
                for(int i=0, bin_index=zeta_index; i<NBIN; i++){

                  alm1w = alm1wlist[i];

                  // Iterate over second bin
                  for(int j=i; j<NBIN; j++){

                    alm2conj = alm2conjlist[j];

                    // Iterate over final bin
                    for(int k=j; k<NBIN; k++, bin_index++){

                      fourpcf[bin_index] += (alm1w*alm3list2[k]*alm2conj).real();
                    }
                  }
                }
              }

              // Use both sets
              if((weight1!=0)&&(weight2!=0)){
                for(int i=0, bin_index=zeta_index; i<NBIN; i++){

                  alm1w = alm1wlist[i];

                  // Iterate over second bin
                  for(int j=i; j<NBIN; j++){

                    alm2 = alm2list[j];
                    alm2conj = alm2conjlist[j];

                    // Iterate over final bin
                    for(int k=j; k<NBIN; k++, bin_index++){

                      // First combination
                      fourpcf[bin_index] += (alm1w*alm2*alm3conjlist1[k]).real() + (alm1w*alm3list2[k]*alm2conj).real();
                    }
                  }
                }
              }
              //End of radial binning loops
            }
          }
        }
      }
    }

// //
// //
// //   // First compute the conjugate of the alm array for later use
// //   Complex almconj[NBIN][NLM];   // Apparently this initializes to zero
// //   for(int i=0;i<NBIN;i++){
// //     for(int j=0;j<NLM;j++){
// //       almconj[i][j] = conj(alm[i][j]);
// //     }
// //   }
// //
// //
// //
// //     Complex alm1w, alm12w, alm123w;
// //     Complex alm1wlist[NBIN], alm2list[NBIN], alm3list[NBIN]; // arrays to hold all alms
// //     Float this_weight;
// //     int m3;
// //
// //
// //  // hold useful quantities to avoid recomputation
// //   int l1tmp,l2tmp,l3tmp;
// //
// //   for (int l1=0, n=0, l_index=0; l1<=ORDER; l1++) {
// //     l1tmp = l1*(l1+1)/2;
// //
// //     for (int l2=0; l2<=ORDER; l2++) {
// //       l2tmp = l2*(l2+1)/2;
// //
// //       for (int l3=fmax(0,fabs(l1-l2)); l3<=fmin(ORDER,l1+l2); l3++, l_index++) {
// //
// //         // don't fill odd parity bins (can't skip these until later, else we'll mess up the index counting)
// //         if(pow(-1.,l1+l2+l3)==-1.){
// //           n+=(2*l1+1)*(2*l2+1); // move to next relevant point in array
// //           continue;
// //         }
// //         l3tmp = l3*(l3+1)/2;
// //
// //         for (int m1=-l1; m1<=l1; m1++) {
// //
// //           // Load list of primary spherical harmonics, taking conjugate if negative (and including primary weight)
// //           // Note that the associated (-1)^m factor has been absorbed into the weight
// //           if (m1<0) for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*almconj[x][l1tmp-m1];
// //           else for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*alm[x][l1tmp+m1];
// //
// //           for (int m2=-l2; m2<=l2; m2++) {
// //             n++;
// //
// //             this_weight = weight4pcf[n];
// //             if (this_weight==0) continue; // zero contribution
// //
// //             m3 = -m1-m2;
// //             if (fabs(m3)>l3) continue; // violates triangle conditions
// //
// //             // Load secondary spherical harmonics
// //             if (m2<0) for(int x=0;x<NBIN;x++) alm2list[x] = almconj[x][l2tmp-m2];
// //             else for(int x=0;x<NBIN;x++) alm2list[x] = alm[x][l2tmp+m2];
// //
// //             // Load tertiary spherical harmonics
// //             if (m3<0) for(int x=0;x<NBIN;x++) alm3list[x] = almconj[x][l3tmp-m3];
// //             else for(int x=0;x<NBIN;x++) alm3list[x] = alm[x][l3tmp+m3];
// //
// //             // Iterate over the hypertriangle of radial fourpcf bins, indexed by bin_index
// //             for (int i=0, bin_index=l_index*N4PCF; i<NBIN; i++) {
// //                 alm1w = alm1wlist[i];
// //                 for (int j=0; j<=i; j++) {
// //                   alm12w = alm1w*alm2list[j];
// //                   for (int k=0; k<=j; k++, bin_index++) {
// //                     alm123w = alm12w*alm3list[k];
// //                     // Accumulate 4PCF contribution
// //                     fourpcf[bin_index] += alm123w.real()*this_weight;
// //             }
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
// // }

  BinTimer4.Stop();

#endif

	return;
    }

};  // end NPCF class

#endif
