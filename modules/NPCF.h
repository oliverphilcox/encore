#ifndef NPCF_H
#define NPCF_H

// ========================== Here are all of the cross-powers ============

#include "WeightFunctions.h"
#ifdef GPU
#include "gpufuncs.h"
#endif

class NPCF {
// This should accumulate the NPCF contributions, for all combination of bins.
  public:
#ifdef GPU
    bool generate_luts4 = true;
    bool generate_luts = true;
    //declare LUTs as pointers -- we'll want to allocate and populate once
    int *lut4_l1, *lut4_l2, *lut4_l3;
    bool *lut4_odd;
    int *lut4_m1, *lut4_m2, *lut4_n, *lut4_zeta;
    int *lut4_i, *lut4_j, *lut4_k;
    double *d_weight4pcf, *d_fourpcf;
    //declare pointers for float operations - only used if -float but
    //simply declaring them doesn't cost much 
    float *f_weight4pcf, *f_fourpcf;

    int *lut5_l1, *lut5_l2, *lut5_l12, *lut5_l3, *lut5_l4;
    bool *lut5_odd;
    int *lut5_m1, *lut5_m2, *lut5_m3;
    int *lut5_n, *lut5_zeta;
    int *lut5_i, *lut5_j, *lut5_k, *lut5_l;
    double *d_weight5pcf, *d_fivepcf;
    //declare pointers for float operations - only used if -float but
    //simply declaring them doesn't cost much
    float *f_weight5pcf, *f_fivepcf;
#endif

    int bincounts[NBIN];
    Float binweight[NBIN];
    int map[MAXORDER+1][MAXORDER+1][MAXORDER+1];   // The multipole index of x^a y^b z^c
    STimer MultTimer;
    STimer BinTimer3;

    // Array to hold the 3PCF
    #define NL (ORDER+1)
    #define N3PCF (NBIN*(NBIN-1)/2) // only compute bin1 < bin2
    Float threepcf[N3PCF*NL];

#ifdef FOURPCF
    STimer BinTimer4;

// Sizes of 4pcf array
#define N4PCF (NBIN*(NBIN-1)*(NBIN-2)/6)

    // Array to hold the 4PCF (some bins will violate triangle condition / parity and be empty
    Float *fourpcf;
    // length of angular part of 4PCF (set at run-time)
    int nell4;
    int nouter4, ninner4;
#endif

#ifdef DISCONNECTED
  STimer BinTimerDisc;

  // Booleans to indicate run-type
  int qbalance, qinvert;

  // Arrays to hold the disconnected 2PCF components
  Complex *discon1;
  Complex *discon2;
#endif

#ifdef FIVEPCF
    STimer BinTimer5;

// Sizes of 5pcf array
#define N5PCF (NBIN*(NBIN-1)*(NBIN-2)*(NBIN-3)/24)

    // Array to hold the 5PCF (including only even-parity, and triangle-condition-satisfying bins)
    Float *fivepcf;
    // length of angular part of 5PCF
    int nell5;
    //lengths of look up tables for indices
    int nouter5, ninner5;
#endif

#ifdef SIXPCF
    STimer BinTimer6;

// Sizes of 5pcf array
#define N6PCF (NBIN*(NBIN-1)*(NBIN-2)*(NBIN-3)*(NBIN-4)/120)

    // Array to hold the 6PCF (including only even-parity, and triangle-condition-satisfying bins)
    Float *sixpcf;
    // length of angular part of 6PCF
    int nell6;
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
	    for (int j=i+1; j<NBIN; j++, ct++) {
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
        for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2);l3++) nell4++;
      }
    }
    // Now allocate memory
    fourpcf = (Float *)malloc(sizeof(Float)*nell4*N4PCF);

    // Initialize array to zero
    for(int x=0;x<nell4*N4PCF;x++) fourpcf[x] = 0.0;
#endif

#ifdef DISCONNECTED
  // Assign memory to the disconnected 2PCF arrays
  discon1 = (Complex *)malloc(sizeof(Complex)*NL*NL*NBIN);
  discon2 = (Complex *)malloc(sizeof(Complex)*NL*NL*NL*NL*N3PCF);

  // Initialize arrays to zero
  for (int x=0;x<NL*NL*NBIN;x++) discon1[x] = {0.,0.};
  for (int x=0;x<NL*NL*NL*NL*N3PCF;x++) discon2[x] = {0.,0.};
#endif

#ifdef FIVEPCF

    // First work out how long the 5PCF array should be, taking into account triangle conditions
    // note we have odd and even parity here (could remove odd)
    // we allow intermediate momenta to have any ell allowed by the triangle conditions (i.e. we don't restrict l12<=ORDER)
    nell5=0;
    for(int l1=0;l1<=ORDER;l1++){
      for(int l2=0;l2<=ORDER;l2++){
        for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
          for(int l3=0;l3<=ORDER;l3++){
            for(int l4=fabs(l12-l3);l4<=fmin(ORDER,l12+l3);l4++) nell5++;
          }
        }
      }
    }

    // Now allocate memory
    fivepcf = (Float *)malloc(sizeof(Float)*nell5*N5PCF);

    // Initialize array to zero
    for(int x=0;x<nell5*N5PCF;x++) fivepcf[x] = 0.0;
#endif


#ifdef SIXPCF

    // First work out how long the 6PCF array should be, taking into account triangle conditions
    // we allow intermediate momenta to have any ell allowed by the triangle conditions (i.e. we don't restrict l12,l123<=ORDER)
    nell6=0;
    for(int l1=0;l1<=ORDER;l1++){
      for(int l2=0;l2<=ORDER;l2++){
        for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
          for(int l3=0;l3<=ORDER;l3++){
            for(int l123=fabs(l12-l3);l123<=l12+l3;l123++){
              for(int l4=0;l4<=ORDER;l4++){
                  for(int l5=fabs(l123-l4);l5<=fmin(ORDER,l123+l4);l5++) nell6++;
              }
            }
          }
        }
      }
    }
    // Now allocate memory
    sixpcf = (Float *)malloc(sizeof(Float)*nell6*N6PCF);

    // Initialize array to zero
    for(int x=0;x<nell6*N6PCF;x++) sixpcf[x] = 0.0;
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

  #ifdef DISCONNECTED
  for(int x=0;x<NL*NL*NBIN;x++) discon1[x] += c->discon1[x];
  for(int x=0;x<NL*NL*NL*NL*N3PCF;x++) discon2[x] += c->discon2[x];
  #endif

  #ifdef FIVEPCF
  for(int x=0;x<N5PCF*nell5;x++) fivepcf[x] += c->fivepcf[x];
  #endif

  #ifdef SIXPCF
  for(int x=0;x<N6PCF*nell6;x++) sixpcf[x] += c->sixpcf[x];
  #endif
    }

    #ifdef DISCONNECTED
    void load_params(int _qbalance, int _qinvert){
      qbalance = _qbalance;
      qinvert = _qinvert;
    }
    #endif

    void report_timings() {
      /// Report the NPCF timing measurements (for a single CPU).

      printf("\n# Single CPU Timings");
      printf("\nSpherical harmonics: %.3f s",MultTimer.Elapsed());
      printf("\n3PCF binning: %.3f s",BinTimer3.Elapsed());
#ifdef FOURPCF
        printf("\n4PCF binning: %.3f s",BinTimer4.Elapsed());
#endif
#ifdef DISCONNECTED
        printf("\nDisconnected 4PCF binning: %.3f s",BinTimerDisc.Elapsed());
#endif
#ifdef FIVEPCF
        printf("\n5PCF binning: %.3f s",BinTimer5.Elapsed());
#endif
#ifdef SIXPCF
        printf("\n6PCF binning: %.3f s",BinTimer6.Elapsed());
#endif
      printf("\n");
    }

    void save_power(char* out_string, Float rmin, Float rmax) {
      // Print the output NPCF counts to file

      // SAVE 3PCF

      // First create output files
       char out_name[1000];
        snprintf(out_name, sizeof out_name, "output/%s_3pcf.txt", out_string);
       FILE * OutFile = fopen(out_name,"w");

       // Print some useful information
       fprintf(OutFile,"## Order: %d\n",ORDER);
       fprintf(OutFile,"## Bins: %d\n",NBIN);
       fprintf(OutFile,"## Minimum Radius = %.2e\n", rmin);
       fprintf(OutFile,"## Maximum Radius = %.2e\n", rmax);
       fprintf(OutFile,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Rows 3+ = zeta_ell^ab\n");
       fprintf(OutFile,"## Column 1 specifies the angular multipole\n");

       // First print the indices of the first radial bin
       fprintf(OutFile," \t"); // empty l1 specifier
       for(int i=0;i<NBIN;i++){
         for(int j=i+1; j<NBIN; j++) fprintf(OutFile,"%2d\t",i);
       }
       fprintf(OutFile," \n");

       // Print the indices of the second radial bin
       fprintf(OutFile,"\t"); // empty ell specifier
       for(int i=0;i<NBIN;i++){
         for(int j=i+1; j<NBIN; j++) fprintf(OutFile,"%2d\t",j);
       }
       fprintf(OutFile,"\n");

       // Now print the 3PCF, ell-by-ell.
       for(int ell=0;ell<=ORDER;ell++){
         fprintf(OutFile,"%d\t",ell);

           for (int i=0, ct=0; i<NBIN; i++){
             for(int j=i+1; j<NBIN; j++, ct++){
                fprintf(OutFile,"%le\t",threepcf[ell*N3PCF+ct]);
              }
          }
          fprintf(OutFile,"\n");
        }

       fflush(NULL);

       // Close open files
       fclose(OutFile);

       printf("3PCF Output saved to %s\n",out_name);

#ifdef FOURPCF
    {
       // SAVE 4PCF

       // First create output files
       char out_name2[1000];
       snprintf(out_name2, sizeof out_name2, "output/%s_4pcf.txt", out_string);
       FILE * OutFile2 = fopen(out_name2,"w");

        // Print some useful information
        fprintf(OutFile2,"## Order: %d\n",ORDER);
        fprintf(OutFile2,"## Bins: %d\n",NBIN);
        fprintf(OutFile2,"## Minimum Radius = %.2e\n", rmin);
        fprintf(OutFile2,"## Maximum Radius = %.2e\n", rmax);
        #ifdef ALLPARITY
          fprintf(OutFile2,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Rows 4+ = zeta_l1l2l3^abc. For odd parity multiplets, we give the value of -i*zeta_l1l2l3^abc.\n");
        #else
          fprintf(OutFile2,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Rows 4+ = zeta_l1l2l3^abc\n");
        #endif
        fprintf(OutFile2,"## Columns 1-3 specify the (l1, l2, l3) multipole triplet\n");

        // First print the indices of the radial bins
        fprintf(OutFile2,"\t\t\t"); // empty l1,l2,l3 specifier
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              fprintf(OutFile2,"%2d\t",i);
            }
          }
        }
        fprintf(OutFile2,"\n");

        fprintf(OutFile2,"\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              fprintf(OutFile2,"%2d\t",j);
            }
          }
        }
        fprintf(OutFile2,"\n");

        fprintf(OutFile2,"\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              fprintf(OutFile2,"%2d\t",k);
            }
          }
        }
        fprintf(OutFile2,"\n");

        // Now print the 4PCF, ell-by-ell.
        for(int l1=0,l_index=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
            for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2);l3++,l_index++){
              #ifndef ALLPARITY
                if(pow(-1.,l1+l2+l3)==-1) continue; // skip odd parity and triangle violating bins
              #endif
              fprintf(OutFile2,"%d\t%d\t%d\t",l1,l2,l3);
              for (int i=0;i<N4PCF;i++) fprintf(OutFile2,"%le\t",fourpcf[N4PCF*l_index+i]);
              fprintf(OutFile2,"\n");
            }
          }
        }
        fflush(NULL);

        // Close open files
        fclose(OutFile2);

        printf("4PCF Output saved to %s\n",out_name2);
      }

#endif

#ifdef DISCONNECTED
       {
       // Save first disconnected piece, i.e. xi_lm(r)

       // First create output files
       char out_name2[1000];
       snprintf(out_name2, sizeof out_name2, "output/%s_2pcf_mult1.txt", out_string);
       FILE * OutFile2 = fopen(out_name2,"w");

        // Print some useful information
        fprintf(OutFile2,"## Order: %d\n",ORDER);
        fprintf(OutFile2,"## Bins: %d\n",NBIN);
        fprintf(OutFile2,"## Minimum Radius = %.2e\n", rmin);
        fprintf(OutFile2,"## Maximum Radius = %.2e\n", rmax);
        fprintf(OutFile2,"## Format: Row 1 = radial bin, Rows 2+ = xi_l1m1^a\n");
        fprintf(OutFile2,"## Columns 1-2 specify the (l1, m1) pair\n");

        // First print the indices of the radial bins
        fprintf(OutFile2,"\t\t"); // empty l1,m1 specifier
        for(int i=0;i<NBIN;i++) fprintf(OutFile2,"%2d\t\t",i);
        fprintf(OutFile2,"\n");

         // Now print the 2PCF harmonics, ell-by-ell.
         for(int l1=0,ct1=0;l1<=ORDER;l1++){
           for(int m1=-l1;m1<=l1;m1++,ct1++){
             fprintf(OutFile2,"%d\t%d\t",l1,m1);
             for (int i=0;i<NBIN;i++) fprintf(OutFile2,"%le\t%le\t",discon1[NBIN*ct1+i].real(),discon1[NBIN*ct1+i].imag());
             fprintf(OutFile2,"\n");
           }
         }

        fflush(NULL);

        // Close open files
        fclose(OutFile2);

        printf("First disconnected piece saved to %s\n",out_name2);
      }

      {
      // Save second disconnected piece, i.e. xi_{l1m1l2m2}(r1,r2)

      // First create output files
      char out_name2[1000];
      snprintf(out_name2, sizeof out_name2, "output/%s_2pcf_mult2.txt", out_string);
      FILE * OutFile2 = fopen(out_name2,"w");

       // Print some useful information
       fprintf(OutFile2,"## Order: %d\n",ORDER);
       fprintf(OutFile2,"## Bins: %d\n",NBIN);
       fprintf(OutFile2,"## Minimum Radius = %.2e\n", rmin);
       fprintf(OutFile2,"## Maximum Radius = %.2e\n", rmax);
       fprintf(OutFile2,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Rows 3+ = xi_l1m1l2m2^{ab}\n");
       fprintf(OutFile2,"## Columns 1-4 specify the (l1, m1, l2, m2) quad\n");

       // First print the indices of the radial bins
       fprintf(OutFile2,"\t\t\t\t"); // empty l1,m1,l2,m2 specifier
       for(int i=0;i<NBIN;i++){
         for(int j=i+1; j<NBIN; j++){
           fprintf(OutFile2,"%2d\t\t",i);
         }
       }
       fprintf(OutFile2,"\n");

       fprintf(OutFile2,"\t\t\t\t"); // empty l1,m1,l2,m2 specifier
       for(int i=0;i<NBIN;i++){
         for(int j=i+1; j<NBIN; j++){
           fprintf(OutFile2,"%2d\t\t",j);
         }
       }
       fprintf(OutFile2,"\n");

       // Now print the 2PCF harmonics, ell-by-ell.
       for(int l1=0,ct=0;l1<=ORDER;l1++){
         for(int m1=-l1;m1<=l1;m1++){
           for(int l2=0;l2<=ORDER;l2++){
             for(int m2=-l2;m2<=l2;m2++,ct++){
               fprintf(OutFile2,"%d\t%d\t%d\t%d\t",l1,m1,l2,m2);
               for (int i=0;i<N3PCF;i++) fprintf(OutFile2,"%le\t%le\t",discon2[N3PCF*ct+i].real(),discon2[N3PCF*ct+i].imag());
               fprintf(OutFile2,"\n");
             }
           }
         }
       }
       fflush(NULL);

       // Close open files
       fclose(OutFile2);

       printf("Second disconnected piece saved to %s\n",out_name2);

     }
#endif

#ifdef FIVEPCF

       // SAVE 5PCF

       // First create output files
       char out_name3[1000];
       snprintf(out_name3, sizeof out_name3, "output/%s_5pcf.txt", out_string);
       FILE * OutFile3 = fopen(out_name3,"w");

        // Print some useful information
        fprintf(OutFile3,"## Order: %d\n",ORDER);
        fprintf(OutFile3,"## Bins: %d\n",NBIN);
        fprintf(OutFile3,"## Minimum Radius = %.2e\n", rmin);
        fprintf(OutFile3,"## Maximum Radius = %.2e\n", rmax);
        #ifdef ALLPARITY
          fprintf(OutFile3,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Row 4 = radial bin 4, Rows 5+ = zeta_l1l2(l12)l3l4^abcd. For odd parity multiplets, we give the value of -i*zeta_l1l2(l12)l3l4^abcd.\n");
        #else
          fprintf(OutFile3,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Row 4 = radial bin 4, Rows 5+ = zeta_l1l2(l12)l3l4^abcd\n");
        #endif
        fprintf(OutFile3,"## Columns 1-5 specify the (l1, l2, (l12), l3, l4) multipole quintuplet\n");

        // First print the indices of the radial bins
        fprintf(OutFile3,"\t\t\t\t"); // empty l1,l2,l12,l3,l4 specifier
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                fprintf(OutFile3,"%2d\t",i);
              }
            }
          }
        }
        fprintf(OutFile3,"\n");

        fprintf(OutFile3,"\t\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                fprintf(OutFile3,"%2d\t",j);
              }
            }
          }
        }
        fprintf(OutFile3,"\n");

        fprintf(OutFile3,"\t\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                fprintf(OutFile3,"%2d\t",k);
              }
            }
          }
        }
        fprintf(OutFile3,"\n");

        fprintf(OutFile3,"\t\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                fprintf(OutFile3,"%2d\t",l);
              }
            }
          }
        }
        fprintf(OutFile3,"\n");

        // Now print the 5PCF, ell-by-ell.
        for(int l1=0,l_index=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
            for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
              for(int l3=0;l3<=ORDER;l3++){
                for(int l4=fabs(l12-l3);l4<=fmin(ORDER,l12+l3);l4++,l_index++){
                  #ifndef ALLPARITY
                    if(pow(-1.,l1+l2+l3+l4)==-1) continue; // skip odd parity and triangle violating bins
                  #endif
                  fprintf(OutFile3,"%d\t%d\t%d\t%d\t%d\t",l1,l2,l12,l3,l4);
                  for (int i=0;i<N5PCF;i++) fprintf(OutFile3,"%le\t",fivepcf[N5PCF*l_index+i]);
                  fprintf(OutFile3,"\n");
                }
              }
            }
          }
        }
        fflush(NULL);

        // Close open files
        fclose(OutFile3);

        printf("5PCF Output saved to %s\n",out_name3);

#endif

#ifdef SIXPCF

       // SAVE 6PCF

       // First create output files
       char out_name4[1000];
       snprintf(out_name4, sizeof out_name4, "output/%s_6pcf.txt", out_string);
       FILE * OutFile4 = fopen(out_name4,"w");

        // Print some useful information
        fprintf(OutFile4,"## Order: %d\n",ORDER);
        fprintf(OutFile4,"## Bins: %d\n",NBIN);
        fprintf(OutFile4,"## Maximum Radius = %.2e\n", rmin);
        fprintf(OutFile4,"## Maximum Radius = %.2e\n", rmax);
        #ifdef ALLPARITY
          fprintf(OutFile4,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Row 4 = radial bin 4, Row 5 = radial bin 5, Rows 6+ = zeta_l1l2(l12)l3(l123)l4l5^abcde. For odd parity multiplets, we give the value of -i*zeta_l1l2(l12)l3(l123)l4l5^abcde.\n");
        #else
          fprintf(OutFile4,"## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Row 4 = radial bin 4, Row 5 = radial bin 5, Rows 6+ = zeta_l1l2(l12)l3(l123)l4l5^abcde.\n");
        #endif
        fprintf(OutFile4,"## Columns 1-7 specify the (l1, l2, (l12), l3, (l123), l4, l5) multipole septuplet\n");

        // First print the indices of the radial bins
        fprintf(OutFile4,"\t\t\t\t\t"); // empty l1,l2,l12,l3,l123,l4,l5 specifier
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                for(int m=l+1; m<NBIN; m++){
                  fprintf(OutFile4,"%2d\t",i);
                }
              }
            }
          }
        }
        fprintf(OutFile4,"\n");

        fprintf(OutFile4,"\t\t\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                for(int m=l+1; m<NBIN; m++){
                  fprintf(OutFile4,"%2d\t",j);
                }
              }
            }
          }
        }
        fprintf(OutFile4,"\n");

        fprintf(OutFile4,"\t\t\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                for(int m=l+1; m<NBIN; m++){
                  fprintf(OutFile4,"%2d\t",k);
                }
              }
            }
          }
        }
        fprintf(OutFile4,"\n");

        fprintf(OutFile4,"\t\t\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                for(int m=l+1; m<NBIN; m++){
                  fprintf(OutFile4,"%2d\t",l);
                }
              }
            }
          }
        }

        fprintf(OutFile4,"\n");
        fprintf(OutFile4,"\t\t\t\t\t");
        for(int i=0;i<NBIN;i++){
          for(int j=i+1; j<NBIN; j++){
            for(int k=j+1; k<NBIN; k++){
              for(int l=k+1; l<NBIN; l++){
                for(int m=l+1; m<NBIN; m++){
                  fprintf(OutFile4,"%2d\t",m);
                }
              }
            }
          }
        }
        fprintf(OutFile4,"\n");

        // Now print the 6PCF, ell-by-ell.
        for(int l1=0,l_index=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
            for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
              for(int l3=0;l3<=ORDER;l3++){
                for(int l123=fabs(l12-l3);l123<=l12+l3;l123++){
                  for(int l4=0;l4<=ORDER;l4++){
                    for(int l5=fabs(l123-l4);l5<=fmin(ORDER,l123+l4);l5++,l_index++){
                      #ifndef ALLPARITY
                        if(pow(-1.,l1+l2+l3+l4+l5)==-1) continue; // skip odd parity and triangle violating bins
                      #endif
                      fprintf(OutFile4,"%d\t%d\t%d\t%d\t%d\t%d\t%d\t",l1,l2,l12,l3,l123,l4,l5);
                      for (int i=0;i<N6PCF;i++) fprintf(OutFile4,"%le\t",sixpcf[N6PCF*l_index+i]);
                      fprintf(OutFile4,"\n");
                    }
                  }
                }
              }
            }
          }
        }
        fflush(NULL);

        // Close open files
        fclose(OutFile4);

        printf("6PCF Output saved to %s\n",out_name4);

#endif

     }

  inline void add_to_power(Multipoles *mult, Float wp) {
      // wp is the primary galaxy weight
	// Now use all of the binned multipoles to compute the
	// spherical harmonics in all bins and then the cross-powers.
	// Need some scratch space:
	// This also applies the weight of the primary galaxy.
        MultTimer.Start();
	Complex alm[NBIN][NLM];   // Apparently this initializes to zero

        //if (_gpumode == 0) {
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
//}

#define RealProduct(a,b) (a.real()*b.real()+a.imag()*b.imag())

        //compute conjugates if not gpu mode
        Complex almconj[NBIN][NLM];
        //if (_gpumode == 0) {
          for(int x=0;x<NBIN;x++){
            for(int l=0, y=0;l<=ORDER;l++){
              for(int m=0;m<=l;m++,y++) almconj[x][y] = conj(alm[x][y]);
            }
          }
        //}
        MultTimer.Stop();

        // COMPUTE 3PCF CONTRIBUTIONS
        // Precompute complex conjugates of all alm (for m>=0)
        BinTimer3.Start();

	for (int i=0, ct=0; i<NBIN; i++) {
	    for (int j=i+1; j<NBIN; j++, ct++) {
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
      			    wp*(alm[i][n]*almconj[j][n]).real()*weight3pcf[n]; //* 4.0 * M_PI / (2*ell+1.0);
          }
        }
      }
    }

  BinTimer3.Stop();

#ifdef DISCONNECTED

    BinTimerDisc.Start();

    // FIRST TERM
    Float weight1,weight2;
    Complex tmp;

    // Iterate over angular bins
  	for (int ell=0, n=0; ell<=ORDER; ell++) {
	    for (int mm=-ell; mm<=ell; mm++, n++) {
        weight1 = wp*weightdiscon[n];
        // Add to array, taking conjugate if necessary
        if (mm<0) for(int i=0; i<NBIN; i++) discon1[n*NBIN+i] += weight1*almconj[i][ell*(ell+1)/2-mm];
        else for (int i=0; i<NBIN; i++) discon1[n*NBIN+i] += weight1*alm[i][ell*(ell+1)/2+mm];
      }
    }

    // SECOND TERM
    // Accumulate only if first particle is a random.
    if (((wp<0)&&(qbalance))||(qinvert)){

      // Iterate over radial bins
      for (int ell1=0, n1=0, ct_ang=0; ell1<=ORDER; ell1++) {
        for (int mm1=-ell1; mm1<=ell1; mm1++, n1++) {
          weight1 = wp*weightdiscon[n1];
          for (int ell2=0, n2=0; ell2<=ORDER; ell2++) {
            for (int mm2=-ell2; mm2<=ell2; mm2++, n2++,ct_ang++){
                weight2 = weight1*weightdiscon[n2];
                if(mm1<0){
                  for(int i=0, ct_rad=0; i<NBIN; i++) {
                    tmp = weight2*almconj[i][ell1*(ell1+1)/2-mm1];
                    if(mm2<0) for(int j=i+1; j<NBIN; j++, ct_rad++) discon2[ct_ang*N3PCF+ct_rad] += tmp*almconj[j][ell2*(ell2+1)/2-mm2];
                    else for(int j=i+1; j<NBIN; j++, ct_rad++) discon2[ct_ang*N3PCF+ct_rad] += tmp*alm[j][ell2*(ell2+1)/2+mm2];
                  }
                }
                else{
                  for(int i=0, ct_rad=0; i<NBIN; i++) {
                    tmp = weight2*alm[i][ell1*(ell1+1)/2+mm1];
                    if(mm2<0) for(int j=i+1; j<NBIN; j++, ct_rad++) discon2[ct_ang*N3PCF+ct_rad] += tmp*almconj[j][ell2*(ell2+1)/2-mm2];
                    else for(int j=i+1; j<NBIN; j++, ct_rad++) discon2[ct_ang*N3PCF+ct_rad] += tmp*alm[j][ell2*(ell2+1)/2+mm2];
                  }
                }
              }
            }
          }
        }
    }
    BinTimerDisc.Stop();

#endif

  #ifdef FOURPCF
      {
      // COMPUTE 4PCF CONTRIBUTIONS

      BinTimer4.Start();

      int tmp_l1, tmp_l2, tmp_l3, tmp_lm3, m3; // useful indices
      #ifdef ALLPARITY
        int odd; // flag to say whether multiplet is odd-parity
      #endif
      Float weight; // coupling weight
      Complex alm1wlist[NBIN], alm2list[NBIN]; // arrays to hold intermediate a_lm lists
      Complex alm1w, alm2; // intermediate a_lm values

#ifdef GPU
    int n; // indexes weight array
    if (_gpumode == 0) generate_luts4 = false;
    if (generate_luts4) {
      //can only get here in _gpumode > 0
      generate_luts4 = false;
      n = 0;
      nouter4 = 0;
      ninner4 = N4PCF;

      //first calculate nouter4 = LUT size
      if (_gpumode == 1) {
        //use primary GPU kernel
        //generate LUTs here for primary GPU kernel
        //ms are looped over in kernel so we have nell4*N4PCF threads
        for(int l1=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
            for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2); l3++){
              // Skip any odd multipoles with odd parity
              #ifndef ALLPARITY
                if(pow(-1,l1+l2+l3)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
              #endif
              nouter4++;
            }
          }
        }
      } else if (_gpumode == 2) {
        //alternate kernel - every inner loop operation is a thread
        //need to create larger LUTs of ls and ms
        for(int l1=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
            for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2); l3++){
              // Skip any odd multipoles with odd parity
              #ifndef ALLPARITY
                if(pow(-1,l1+l2+l3)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
              #endif
              for(int m1=-l1; m1<=l1; m1++){
                // Iterate over all m2 (including negative)
                for(int m2=-l2; m2<=l2; m2++){
                  m3 = -m1-m2;
                  if (m3<0) continue; // only need to use m3>=0
                  if (m3>l3) continue; // this violates triangle conditions
                  // Look up the relevant weight
                  weight = weight4pcf[n++];
                  if (weight==0) continue;
                  nouter4++;
                }
              }
            }
          }
        }
      }

      //malloc all look up tables (LUTs) - primary kernel
      gpu_allocate_luts4(&lut4_l1, &lut4_l2, &lut4_l3, &lut4_odd, &lut4_n,
	&lut4_zeta, &lut4_i, &lut4_j, &lut4_k, nouter4, ninner4);

      if (_gpumode == 2) {
        //malloc m LUTs
        gpu_allocate_m_luts4(&lut4_m1, &lut4_m2, nouter4);
      }

      //if not using memcpy, allocate pointers on device here
      int size_w = (ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1);
      if (_gpufloat) {
        //allocate float arrays and cast as floats when copying
        gpu_allocate_weight4pcf(&f_weight4pcf, weight4pcf, size_w);
        gpu_allocate_fourpcf(&f_fourpcf, fourpcf, nell4*N4PCF);
      } else {
        //normal mode - allocate GPU arrays and copy
        gpu_allocate_weight4pcf(&d_weight4pcf, weight4pcf, size_w);
        gpu_allocate_fourpcf(&d_fourpcf, fourpcf, nell4*N4PCF);
      }

      //populate LUTs
      int iouter4 = 0;
      n = 0; //DONT FORGET TO RESET N!

      if (_gpumode == 1) {
        //use primary GPU kernel
        //generate LUTs here for primary GPU kernel
        //ms are looped over in kernel so we have nell4*N4PCF threads
        for(int l1=0,zeta_index=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
	    for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2); l3++, zeta_index+=N4PCF){
              #ifndef ALLPARITY
                // Skip any odd multipoles with odd parity
	        if(pow(-1,l1+l2+l3)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
                lut4_odd[iouter4] = false;
              #else
                // Store which variables are odd!
                if(pow(-1,l1+l2+l3)==-1) lut4_odd[iouter4]=true; else lut4_odd[iouter4]=false;
              #endif
              //update l luts here
              lut4_l1[iouter4] = l1;
              lut4_l2[iouter4] = l2;
              lut4_l3[iouter4] = l3;
              lut4_n[iouter4] = n; //this is the starting n for this GPU thread
              //GPU thread will then loop over ms
              lut4_zeta[iouter4] = zeta_index;
              //loop over ms - need to update n so that LUT has correct
              //n_init for all threads 
              for(int m1=-l1; m1<=l1; m1++){
                // Iterate over all m2 (including negative)
                for(int m2=-l2; m2<=l2; m2++){
                  m3 = -m1-m2;
                  if (m3<0) continue; // only need to use m3>=0
                  if (m3>l3) continue; // this violates triangle conditions
                  //simply increment n here
                  n++;
                }
              }
              iouter4++; //now increment iouter4
            }
          }
        }
      } else if (_gpumode == 2) {
        //use alternate kernel
        //need to create larger LUTs of ls and ms
        for(int l1=0,zeta_index=0;l1<=ORDER;l1++){
          for(int l2=0;l2<=ORDER;l2++){
            for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2); l3++, zeta_index+=N4PCF){
              #ifndef ALLPARITY
                // Skip any odd multipoles with odd parity
                if(pow(-1,l1+l2+l3)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
              #endif
              //loop over ms - need to update n so that LUT has correct
              for(int m1=-l1; m1<=l1; m1++){
                // Iterate over all m2 (including negative)
                for(int m2=-l2; m2<=l2; m2++){
                  m3 = -m1-m2;
                  if (m3<0) continue; // only need to use m3>=0
                  if (m3>l3) continue; // this violates triangle conditions
                  // Look up the relevant weight
                  weight = weight4pcf[n++];
                  if (weight==0) continue;

                  #ifndef ALLPARITY
                    lut4_odd[iouter4] = false;
                  #else
                    if(pow(-1,l1+l2+l3)==-1) lut4_odd[iouter4]=true; else lut4_odd[iouter4]=false;
                  #endif
                  lut4_l1[iouter4] = l1;
                  lut4_l2[iouter4] = l2;
                  lut4_l3[iouter4] = l3;
                  lut4_m1[iouter4] = m1;
                  lut4_m2[iouter4] = m2;
                  lut4_n[iouter4] = n-1;
                  lut4_zeta[iouter4] = zeta_index;
                  iouter4++;
                }
              }
            }
          }
        }
      }

      //inner LUTs the same for all kernels
      int iinner = 0;
      for(int i=0; i<NBIN; i++){
        for(int j=i+1; j<NBIN; j++){
          for(int k=j+1; k<NBIN; k++){
            lut4_i[iinner] = i;
            lut4_j[iinner] = j;
            lut4_k[iinner] = k;
            iinner++;
          }
        }
      }
    }

    // Iterate over first multipole
    if (_gpumode == 1) {
      //execute GPU kernel
      if (_gpufloat) {
        //float kernel
        gpu_add_to_power4_float(f_fourpcf, f_weight4pcf,
        	lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_n,
        	lut4_zeta, lut4_i, lut4_j, lut4_k,
        	(float)wp, NBIN, NLM, nouter4, ninner4, nell4);
      } else if (_gpumixed) {
        gpu_add_to_power4_mixed(d_fourpcf, d_weight4pcf,
                lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_n,
                lut4_zeta, lut4_i, lut4_j, lut4_k,
                (float)wp, NBIN, NLM, nouter4, ninner4, nell4);
      } else {
        gpu_add_to_power4(d_fourpcf, d_weight4pcf,
        	lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_n,
        	lut4_zeta, lut4_i, lut4_j, lut4_k,
        	wp, NBIN, NLM, nouter4, ninner4, nell4);
      }
    } else if (_gpumode == 2) {
      //execute alternate GPU kernel 
      if (_gpufloat) {
        gpu_add_to_power4_orig_float(f_fourpcf, f_weight4pcf,
                lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_m1,
		lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
                (float)wp, NBIN, NLM, nouter4, ninner4, nell4);
      } else if (_gpumixed) {
        gpu_add_to_power4_orig_mixed(d_fourpcf, d_weight4pcf,
                lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_m1,
                lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
                (float)wp, NBIN, NLM, nouter4, ninner4, nell4);
      } else {
        gpu_add_to_power4_orig(d_fourpcf, d_weight4pcf,
                lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_m1,
                lut4_m2, lut4_n, lut4_zeta, lut4_i, lut4_j, lut4_k,
                wp, NBIN, NLM, nouter4, ninner4, nell4);
      }
    } else if (_gpumode == 0) {
#endif
      // Iterate over (l1, l2, l3) triplet
      // NB: n indexes position in the 4PCF weight array, and must be carefully set
      // We only compute terms with even parity i.e. even l1+l2+l3. These are all real.
      // The odd parity terms could be included if necessary and are purely imaginary

      // Iterate over first multipole
      for(int l1=0, zeta_index=0, n=0; l1<=ORDER; l1++) {
        tmp_l1 = l1*(l1+1)/2;
        // Iterate over second multipole
        for(int l2=0; l2<=ORDER; l2++){
          tmp_l2 = l2*(l2+1)/2;

          // Iterate over internal multipole, avoiding bins violating triangle condition
          for(int l3=fabs(l1-l2);l3<=fmin(ORDER,l1+l2); l3++, zeta_index+=N4PCF){
            #ifdef ALLPARITY
              if(pow(-1,l1+l2+l3)==-1) odd=1; else odd=0;
            #else
              // Skip any odd multipoles with odd parity
              if(pow(-1,l1+l2+l3)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
            #endif
            tmp_l3 = l3*(l3+1)/2;

            // Iterate over all m1 (including negative)
            for(int m1=-l1; m1<=l1; m1++){
              // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
              if (m1<0) for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*almconj[x][tmp_l1-m1];
              else for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*alm[x][tmp_l1+m1];

              // Iterate over all m2 (including negative)
              for(int m2=-l2; m2<=l2; m2++){
                m3 = -m1-m2;
                if (m3<0) continue; // only need to use m3>=0
                if (m3>l3) continue; // this violates triangle conditions
                tmp_lm3 = tmp_l3+m3;

                // Look up the relevant weight and advance index
                weight = weight4pcf[n++];
                if (weight==0) continue;

                // Create temporary copy of a_l2m2 and a_l3m3, taking conjugate if necessary
                // No conjugates needed for a_l3m3 since we fixed m3>=0!
                // Note we add the coupling weight factor to a_l3m3
                if (m2<0) for(int x=0;x<NBIN;x++) alm2list[x] = almconj[x][tmp_l2-m2];
                else for(int x=0;x<NBIN;x++) alm2list[x] = alm[x][tmp_l2+m2];

                // Now fill up the 4PCF.
                // Iterate over first radial bin in lower hypertriangle
                for(int i=0, bin_index=zeta_index; i<NBIN; i++){
                  alm1w = alm1wlist[i];

                  // Iterate over second bin
                  for(int j=i+1; j<NBIN; j++){
                    alm2 = alm2list[j]*alm1w;

                    // Iterate over final bin and advance the 4PCF array counter
                    for(int k=j+1; k<NBIN; k++){

                      // Add contribution to 4PCF array
                      #ifdef ALLPARITY
                        if(odd) fourpcf[bin_index++] += weight*(alm2*alm[k][tmp_lm3]).imag();
                        else fourpcf[bin_index++] += weight*(alm2*alm[k][tmp_lm3]).real();
                      #else
                        fourpcf[bin_index++] += weight*(alm2*alm[k][tmp_lm3]).real();
                      #endif
                    }
                  }
                }
                //End of radial binning loops
              }
            }
          }
        }
      }
//End of l loops
#ifdef GPU
    }
#endif
  BinTimer4.Stop();
  }

  #endif

  #ifdef FIVEPCF
  {
  // COMPUTE 5PCF CONTRIBUTIONS

  BinTimer5.Start();

  int n; // indexes weight array
  int tmp_l1, tmp_l2, tmp_l3, tmp_l4, tmp_lm4, m4; // useful indices
  Float weight; // coupling weights
  Complex alm1wlist[NBIN], alm2list[NBIN], alm3list[NBIN]; // arrays to hold intermediate a_lm lists
  Complex alm1w, alm2, alm3; // intermediate a_lm values
  #ifdef ALLPARITY
    int odd; // flag to see if multiplet is odd-parity
  #endif

#ifdef GPU
  if (_gpumode == 0) generate_luts = false;
  if (generate_luts) {
    //can only get here in _gpumode > 0
    generate_luts = false;
    n = 0;
    nouter5 = 0;
    ninner5 = N5PCF;

    //first calculate nouter5 = LUT size
    if (_gpumode == 1) {
      //use primary GPU kernel
      //generate LUTs here for primary GPU kernel
      //ms are looped over in kernel so we have nell5*N5PCF threads
      for(int l1=0;l1<=ORDER;l1++){
        for(int l2=0;l2<=ORDER;l2++){
          for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
            for(int l3=0;l3<=ORDER;l3++){
              for(int l4=fabs(l12-l3);l4<=fmin(ORDER,l12+l3);l4++) {
                // Skip any odd multipoles with odd parity
                #ifndef ALLPARITY
                  if(pow(-1,l1+l2+l3+l4)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
                #endif
	        nouter5++;
              }
            }
          }
        }
      }
    } else if (_gpumode == 2) {
      //alternate kernel - every inner loop operation is a thread
      //need to create larger LUTs of ls and ms
      for(int l1=0;l1<=ORDER;l1++){
        for(int l2=0;l2<=ORDER;l2++){
          for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
            for(int l3=0;l3<=ORDER;l3++){
              for(int l4=fabs(l12-l3);l4<=fmin(ORDER,l12+l3);l4++) {
                // Skip any odd multipoles with odd parity
                #ifndef ALLPARITY
                  if(pow(-1,l1+l2+l3+l4)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
                #endif
                for(int m1=-l1; m1<=l1; m1++){
                  // Iterate over all m2 (including negative)
                  for(int m2=-l2; m2<=l2; m2++){
                    if(abs(m1+m2)>l12) continue; // m12 condition
                    // Iterate over m3 (including negative)
                    for(int m3=-l3; m3<=l3; m3++){
                      m4 = -m1-m2-m3;
                      if (m4<0) continue; // only need to use m4>=0
                      if (m4>l4) continue; // this violates triangle conditions
                      // Look up the relevant weight
                      weight = weight5pcf[n++];
                      if (weight==0) continue;
		      nouter5++;
		    }
	          }
	        }
              }
            }
          }
        }
      }
    }

    //malloc all look up tables (LUTs) - primary kernel
    gpu_allocate_luts(&lut5_l1, &lut5_l2, &lut5_l12, &lut5_l3,
	&lut5_l4, &lut5_odd, &lut5_n, &lut5_zeta, &lut5_i, &lut5_j,
	&lut5_k, &lut5_l, nouter5, ninner5);

    if (_gpumode == 2) {
      //malloc m LUTs
      gpu_allocate_m_luts(&lut5_m1, &lut5_m2, &lut5_m3, nouter5);
    }
    //if not using memcpy, allocate pointers on device here
    int size_w = (ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1)*(2*ORDER+1)*(ORDER+1)*(ORDER+1)*(ORDER+1);
    if (_gpufloat) {
      //allocate float arrays and cast as floats when copying
      gpu_allocate_weight5pcf(&f_weight5pcf, weight5pcf, size_w);
      gpu_allocate_fivepcf(&f_fivepcf, fivepcf, nell5*N5PCF);
    } else { 
      //normal mode - allocate GPU arrays and copy
      gpu_allocate_weight5pcf(&d_weight5pcf, weight5pcf, size_w); 
      gpu_allocate_fivepcf(&d_fivepcf, fivepcf, nell5*N5PCF);
    }

    //populate LUTs
    int iouter5 = 0;
    n = 0; //DONT FORGET TO RESET N!

    if (_gpumode == 1) {
      //use primary GPU kernel
      //generate LUTs here for primary GPU kernel
      //ms are looped over in kernel so we have nell5*N5PCF threads
      for(int l1=0,zeta_index=0;l1<=ORDER;l1++){
        for(int l2=0;l2<=ORDER;l2++){
          for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
            for(int l3=0;l3<=ORDER;l3++){
              for(int l4=fabs(l12-l3); l4<=fmin(ORDER,l12+l3); l4++, zeta_index+=N5PCF){
                // Skip any odd multipoles with odd parity
                #ifndef ALLPARITY
                  if(pow(-1,l1+l2+l3+l4)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
                  lut5_odd[iouter5] = false;
                #else
                  if(pow(-1,l1+l2+l3+l4)==-1) lut5_odd[iouter5] = true; else lut5_odd[iouter5] = false;
                #endif
	        //update l luts here
                lut5_l1[iouter5] = l1;
                lut5_l2[iouter5] = l2;
                lut5_l12[iouter5] = l12;
                lut5_l3[iouter5] = l3;
                lut5_l4[iouter5] = l4;
	        lut5_n[iouter5] = n; //this is the starting n for this GPU thread
	        //GPU thread will then loop over ms
                lut5_zeta[iouter5] = zeta_index;
	        //loop over ms - need to update n so that LUT has correct
	        //n_init for all threads 
                for(int m1=-l1; m1<=l1; m1++){
                  // Iterate over all m2 (including negative)
                  for(int m2=-l2; m2<=l2; m2++){
                    if(abs(m1+m2)>l12) continue; // m12 condition
                    // Iterate over m3 (including negative)
                    for(int m3=-l3; m3<=l3; m3++){
                      m4 = -m1-m2-m3;
                      if (m4<0) continue; // only need to use m4>=0
                      if (m4>l4) continue; // this violates triangle conditions
		      //simply increment n here
		      n++;
                    }
                  }
                }
	        iouter5++; //now increment iouter5
              }
            }
          }
        }
      }
    } else if (_gpumode == 2) {
      //use alternate kernel
      //need to create larger LUTs of ls and ms
      for(int l1=0,zeta_index=0;l1<=ORDER;l1++){
        for(int l2=0;l2<=ORDER;l2++){
          for(int l12=fabs(l1-l2);l12<=l1+l2;l12++){
            for(int l3=0;l3<=ORDER;l3++){
              for(int l4=fabs(l12-l3); l4<=fmin(ORDER,l12+l3); l4++, zeta_index+=N5PCF){
                // Skip any odd multipoles with odd parity
                #ifndef ALLPARITY
                  if(pow(-1,l1+l2+l3+l4)==-1) continue; // nb: these are also skipped in the weights matrix, so no need to update n
                #endif
                //loop over ms - need to update n so that LUT has correct
                for(int m1=-l1; m1<=l1; m1++){
                  // Iterate over all m2 (including negative)
                  for(int m2=-l2; m2<=l2; m2++){
                    if(abs(m1+m2)>l12) continue; // m12 condition
                    // Iterate over m3 (including negative)
                    for(int m3=-l3; m3<=l3; m3++){
                      m4 = -m1-m2-m3;
                      if (m4<0) continue; // only need to use m4>=0
                      if (m4>l4) continue; // this violates triangle conditions
                      // Look up the relevant weight
                      weight = weight5pcf[n++];
                      if (weight==0) continue;

                      #ifndef ALLPARITY
                        lut5_odd[iouter5]=false;
                      #else
                        if(pow(-1,l1+l2+l3+l4)==-1) lut5_odd[iouter5]=true; else lut5_odd[iouter5]=false;
                      #endif

                      lut5_l1[iouter5] = l1;
                      lut5_l2[iouter5] = l2;
                      //lut5_l12[iouter5] = l12; Don't need l12 for alternate kernel
                      lut5_l3[iouter5] = l3;
                      lut5_l4[iouter5] = l4;
                      lut5_m1[iouter5] = m1;
                      lut5_m2[iouter5] = m2;
                      lut5_m3[iouter5] = m3;
                      lut5_n[iouter5] = n-1;
                      lut5_zeta[iouter5] = zeta_index;
		      iouter5++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    //inner LUTs the same for all kernels
    int iinner = 0;
    for(int i=0; i<NBIN; i++){
      for(int j=i+1; j<NBIN; j++){
        for(int k=j+1; k<NBIN; k++){
          for(int l=k+1; l<NBIN; l++){
	    lut5_i[iinner] = i;
            lut5_j[iinner] = j;
            lut5_k[iinner] = k;
            lut5_l[iinner] = l;
	    iinner++;
	  }
	}
      }
    }
  }

  // Iterate over (l1, l2, (l12), l3, l4) quintuplet
  // NB: n indexes position in the 5PCF weight array, and must be carefully set
  // If ALLPARITY is not set, we only compute terms with even parity i.e. even l1+l2+l3+l4. These are all real.
  // Else, we also compute odd-parity terms. These are purely imaginary, and we store only the imaginary part.

  // Iterate over first multipole
  if (_gpumode == 1) {
    //execute GPU kernel
    if (_gpufloat) {
      //float kernel
      gpu_add_to_power5_float(f_fivepcf, f_weight5pcf,
        lut5_l1, lut5_l2, lut5_l12, lut5_l3, lut5_l4, lut5_odd,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        (float)(wp), NBIN, NLM, nouter5, ninner5, nell5);
    } else if (_gpumixed) {
      gpu_add_to_power5_mixed(d_fivepcf, d_weight5pcf,
        lut5_l1, lut5_l2, lut5_l12, lut5_l3, lut5_l4, lut5_odd,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        (float)wp, NBIN, NLM, nouter5, ninner5, nell5);
    } else {
      gpu_add_to_power5(d_fivepcf, d_weight5pcf,
	lut5_l1, lut5_l2, lut5_l12, lut5_l3, lut5_l4, lut5_odd,
	lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
	wp, NBIN, NLM, nouter5, ninner5, nell5);
    }
  } else if (_gpumode == 2) {
    //execute alternate GPU kernel 
    if (_gpufloat) {
      gpu_add_to_power5_orig_float(f_fivepcf, f_weight5pcf,
        lut5_l1, lut5_l2, lut5_l3, lut5_l4, lut5_odd,
        lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        (float)wp, NBIN, NLM, nouter5, ninner5, nell5);
    } else if (_gpumixed) {
      gpu_add_to_power5_orig_mixed(d_fivepcf, d_weight5pcf,
        lut5_l1, lut5_l2, lut5_l3, lut5_l4, lut5_odd,
        lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        (float)wp, NBIN, NLM, nouter5, ninner5, nell5);
    } else {
      gpu_add_to_power5_orig(d_fivepcf, d_weight5pcf,
	lut5_l1, lut5_l2, lut5_l3, lut5_l4, lut5_odd,
	lut5_m1, lut5_m2, lut5_m3,
        lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l,
        wp, NBIN, NLM, nouter5, ninner5, nell5);
    }
  } else if (_gpumode == 0) {
#endif
    //run on CPU
    n=0;
    for(int l1=0, zeta_index=0; l1<=ORDER; l1++) {
      tmp_l1 = l1*(l1+1)/2;
      // Iterate over second multipole
      for(int l2=0; l2<=ORDER; l2++){
        tmp_l2 = l2*(l2+1)/2;
        // Iterate over internal multipole, avoiding bins violating triangle condition
        // but allowing for any allowed internal momenta
        for(int l12=fabs(l1-l2);l12<=l1+l2; l12++){
          // Iterate over third multipole
          for(int l3=0; l3<=ORDER; l3++){
            tmp_l3 = l3*(l3+1)/2;
            // Iterate over fourth multipole, avoiding bins violating triangle condition
            for(int l4=fabs(l12-l3); l4<=fmin(ORDER,l12+l3); l4++, zeta_index+=N5PCF){
              #ifdef ALLPARITY
                if(pow(-1,l1+l2+l3+l4)==-1) odd=1; else odd=0;
              #else
                // Skip any odd multipoles with odd parity
                if(pow(-1,l1+l2+l3+l4)==-1) continue;
              #endif

              tmp_l4 = l4*(l4+1)/2;

              // Iterate over all m1 (including negative)
              for(int m1=-l1; m1<=l1; m1++){
                // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
                if (m1<0) for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*almconj[x][tmp_l1-m1];
                else for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*alm[x][tmp_l1+m1];

                // Iterate over all m2 (including negative)
                for(int m2=-l2; m2<=l2; m2++){
                  if(abs(m1+m2)>l12) continue; // m12 condition
                  // Create temporary copy of a_l2m2, taking conjugate if necessary
                  if (m2<0) for(int x=0;x<NBIN;x++) alm2list[x] = almconj[x][tmp_l2-m2];
                  else for(int x=0;x<NBIN;x++) alm2list[x] = alm[x][tmp_l2+m2];

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
                    if (m3<0) for(int x=0;x<NBIN;x++) alm3list[x] = almconj[x][tmp_l3-m3];
                    else for(int x=0;x<NBIN;x++) alm3list[x] = alm[x][tmp_l3+m3];

                    // Now fill up the 5PCF.
                    // Iterate over first radial bin in lower hypertriangle
                    for(int i=0, bin_index=zeta_index; i<NBIN; i++){
                      alm1w = alm1wlist[i];
                      // Iterate over second bin
                      for(int j=i+1; j<NBIN; j++){
                        alm2 = alm2list[j]*alm1w;
                        // Iterate over third bin
                        for(int k=j+1; k<NBIN; k++){
                          alm3 = alm3list[k]*alm2;
                          // Iterate over final bin and advance the 5PCF array counter
                          for(int l=k+1; l<NBIN; l++){
                            // Add contribution to 5PCF array
                            #ifdef ALLPARITY
                              if(odd) fivepcf[bin_index++] += weight*(alm3*alm[l][tmp_lm4]).imag();
                              else fivepcf[bin_index++] += weight*(alm3*alm[l][tmp_lm4]).real();
                            #else
                              fivepcf[bin_index++] += weight*(alm3*alm[l][tmp_lm4]).real();
                            #endif
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
        }
      }
    }
    //End of l loops
#ifdef GPU
  }
#endif
  BinTimer5.Stop();
}

#endif


#ifdef SIXPCF
  {
  // COMPUTE 6PCF CONTRIBUTIONS

  BinTimer6.Start();

  int n; // indexes weight array
  int tmp_l1, tmp_l2, tmp_l3, tmp_l4, tmp_l5, tmp_lm5, m5; // useful indices
  Float weight; // coupling weights
  Complex alm1wlist[NBIN], alm2list[NBIN], alm3list[NBIN], alm4list[NBIN]; // arrays to hold intermediate a_lm lists
  Complex alm1w, alm2, alm3, alm4; // intermediate a_lm values
  #ifdef ALLPARITY
    int odd; // flag to see whether multiplet has odd-parity
  #endif

  // Iterate over (l1, l2, (l12), l3, (l123), l4, l5) septuplet
  // NB: n indexes position in the 6PCF weight array, and must be carefully set
  // If ALLPARITY is not set, we only compute terms with even parity i.e. even l1+l2+l3+l4+l5. These are all real.
  // Else, we also compute odd parity term. These are purely imaginary and we store only the imaginary part.

  // Iterate over first multipole
  n=0;
  for(int l1=0, zeta_index=0; l1<=ORDER; l1++) {
     tmp_l1 = l1*(l1+1)/2;

     // Iterate over second multipole
     for(int l2=0; l2<=ORDER; l2++){
       tmp_l2 = l2*(l2+1)/2;

       // Iterate over first internal multipole, avoiding bins violating triangle condition
       // NB: we allow this to take any value allowed by the 3j conditions, not just <= ORDER
       for(int l12=fabs(l1-l2);l12<=l1+l2; l12++){

         // Iterate over third multipole
         for(int l3=0; l3<=ORDER; l3++){
           tmp_l3 = l3*(l3+1)/2;

           // Iterate over second internal multipole, avoiding bins violating triangle condition
           // NB: we allow this to take any value allowed by the 3j conditions, not just <= ORDER
           for(int l123=fabs(l12-l3);l123<=l12+l3; l123++){

             // Iterate over fourth multipole, avoiding bins violating triangle condition
             for(int l4=0; l4<=ORDER; l4++){
               tmp_l4 = l4*(l4+1)/2;

               // Iterate over fifth multipole, avoiding bins violating triangle condition
               for(int l5=fabs(l123-l4); l5<=fmin(ORDER,l123+l4); l5++, zeta_index+=N6PCF){
                   #ifdef ALLPARITY
                     if(pow(-1,l1+l2+l3+l4+l5)==-1) odd=1; else odd=0;
                   #else
                     // Skip any odd multipoles with odd parity
                     if(pow(-1,l1+l2+l3+l4+l5)==-1) continue;
                   #endif

                   tmp_l5 = l5*(l5+1)/2;

                   // Iterate over all m1 (including negative)
                   for(int m1=-l1; m1<=l1; m1++){

                     // Create temporary copy of primary_weight*a_l1m1, taking conjugate if necessary [(-1)^m factor is absorbed into weight]
                     if (m1<0) for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*almconj[x][tmp_l1-m1];
                     else for(int x=0;x<NBIN;x++) alm1wlist[x] = wp*alm[x][tmp_l1+m1];

                     // Iterate over all m2 (including negative)
                     for(int m2=-l2; m2<=l2; m2++){
                       if(abs(m1+m2)>l12) continue; // m12 condition

                       // Create temporary copy of a_l2m2, taking conjugate if necessary
                       if (m2<0) for(int x=0;x<NBIN;x++) alm2list[x] = almconj[x][tmp_l2-m2];
                       else for(int x=0;x<NBIN;x++) alm2list[x] = alm[x][tmp_l2+m2];

                       // Iterate over m3 (including negative)
                      for(int m3=-l3; m3<=l3; m3++){
                        if(abs(m1+m2+m3)>l123) continue;

                        // Create temporary copy of a_l3m3, taking conjugate if necessary
                        if (m3<0) for(int x=0;x<NBIN;x++) alm3list[x] = almconj[x][tmp_l3-m3];
                        else for(int x=0;x<NBIN;x++) alm3list[x] = alm[x][tmp_l3+m3];

                        // Iterate over m4 (including negative)
                       for(int m4=-l4; m4<=l4; m4++){

                          m5 = -m1-m2-m3-m4;
                          if (m5<0) continue; // only need to use m5>=0
                          if (m5>l5) continue; // this violates triangle conditions

                          // Look up the relevant weight and advance array
                          weight = weight6pcf[n++];
                          if (weight==0) continue;

                          tmp_lm5 = tmp_l5+m5;

                          // Create temporary copies of a_l4m4 and a_l5m5, taking conjugates if necessary
                          // No conjugates needed for a_l5m5 since we fixed m5>=0!
                          // Note we add the coupling weight factor to a_l5m5
                          if (m4<0) for(int x=0;x<NBIN;x++) alm4list[x] = almconj[x][tmp_l4-m4];
                          else for(int x=0;x<NBIN;x++) alm4list[x] = alm[x][tmp_l4+m4];

                          // Now fill up the 6PCF.
                          // Iterate over first radial bin in lower hypertriangle
                          for(int i=0, bin_index=zeta_index; i<NBIN; i++){

                            alm1w = alm1wlist[i];

                            // Iterate over second bin
                            for(int j=i+1; j<NBIN; j++){

                              alm2 = alm2list[j]*alm1w;

                              // Iterate over third bin
                              for(int k=j+1; k<NBIN; k++){

                                alm3 = alm3list[k]*alm2;

                                // Iterate over fourth bin
                                for(int l=k+1; l<NBIN; l++){

                                  alm4 = alm4list[l]*alm3;

                                  // Iterate over final bin and advance the 6PCF array counter
                                  for(int m=l+1; m<NBIN; m++){
                                      // Add contribution to 6PCF array
                                      #ifdef ALLPARITY
                                        if(odd) sixpcf[bin_index++] += weight*(alm4*alm[m][tmp_lm5]).imag();
                                        else sixpcf[bin_index++] += weight*(alm4*alm[m][tmp_lm5]).real();
                                      #else
                                        sixpcf[bin_index++] += weight*(alm4*alm[m][tmp_lm5]).real();
                                      #endif
                                    }
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
                }
              }
            }
          }
        }
      }
  BinTimer6.Stop();
}

#endif

	return;
    }

    void free_gpu_memory() {
#ifdef GPU
      gpu_free_luts4(lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_n,
	lut4_zeta, lut4_i, lut4_j, lut4_k);
      gpu_free_luts(lut5_l1, lut5_l2, lut5_l12, lut5_l3, lut5_l4, lut5_odd,
	lut5_n, lut5_zeta, lut5_i, lut5_j, lut5_k, lut5_l);
      if (_gpufloat) {
        gpu_free_memory4(f_fourpcf, f_weight4pcf);
        gpu_free_memory(f_fivepcf, f_weight5pcf);
      } else {
        gpu_free_memory4(d_fourpcf, d_weight4pcf);
        gpu_free_memory(d_fivepcf, d_weight5pcf);
      }
      gpu_free_memory_m4(lut4_m1, lut4_m2);
      gpu_free_memory_m(lut5_m1, lut5_m2, lut5_m3);
      gpu_free_memory_alms(!_gpufloat && !_gpumixed);
#endif
    }

    void do_copy_memory() {
#ifdef GPU
#ifdef FOURPCF
      if (_gpufloat) {
        copy_fourpcf(&f_fourpcf, fourpcf, nell4*N4PCF);
      } else {
        copy_fourpcf(&d_fourpcf, fourpcf, nell4*N4PCF);
      }
#endif

#ifdef FIVEPCF
      if (_gpufloat) {
        copy_fivepcf(&f_fivepcf, fivepcf, nell5*N5PCF);
      } else {
        copy_fivepcf(&d_fivepcf, fivepcf, nell5*N5PCF);
      }
#endif
#endif
    }

};  // end NPCF class

#endif
