#ifndef NPCF_H
#define NPCF_H

// ========================== Here are all of the cross-powers ============

#include "WeightFunctions.h"

class NPCF {
// This should accumulate the NPCF contributions, for all combination of bins.
  public:
    uint64 bincounts[NBIN];
    Float binweight[NBIN];
    // Array to hold the 3PCF
    Float threepcf[NBIN][NBIN][ORDER+1];
    int map[MAXORDER+1][MAXORDER+1][MAXORDER+1];   // The multipole index of x^a y^b z^c

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
	for (int i=0; i<NBIN; i++) {
	    bincounts[i] = 0;
	    binweight[i] = 0.0;
	    for (int j=0; j<NBIN; j++) {
		for (int k=0; k<=ORDER; k++) {
		    threepcf[i][j][k] = 0.0;
		}
	    }
	}
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
	    for (int j=0; j<NBIN; j++) {
		for (int k=0; k<=ORDER; k++) {
		    threepcf[i][j][k] += c->threepcf[i][j][k];
		}
	    }
	}
    }

    void report_power() {
      /// Report the NPCF measurements.
      // NB: we print zeta_ij[ell] for all j<=i
      // Old versions printed zeta_ij[ell]/zeta_ij[0]
	for (int i=0; i<NBIN; i++) {
	    for (int j=i; j>=0; j--) {
		if (j==i) printf("Multipole Power %2d %9lld %9.0f\n",
				j, bincounts[j], binweight[j]);
		//if (j==i) {
		//    printf("# Bin auto-power omitted due to uncorrected noise bias\n");
		//    continue;
		//}
		printf("%2d %2d", i, j);
		for (int k=0; k<=ORDER; k++) {
		    printf(" %13.6e", threepcf[i][j][k]); ///threepcf[i][j][0]);
		}
	    printf("\n");
	    }
	}
    }

    void save_power(char* out_string, Float rmax) {
      // Print the output NPCF counts to file

      // First create output files
       char out_name[1000];
        snprintf(out_name, sizeof out_name, "output/%s_3pcf.txt", out_string);
       FILE * OutFile = fopen(out_name,"w");

       // Print some useful information
       fprintf(OutFile,"## Order: %d\n",ORDER);
       fprintf(OutFile,"## Bins: %d\n",NBIN);
       fprintf(OutFile,"## Maximum Radius = %.2e\n", rmax);

       fprintf(OutFile,"Bin 1\tBin 2\t");
       for(int ell=0;ell<=ORDER;ell++) fprintf(OutFile,"zeta_%d\t",ell);
       fprintf(OutFile,"\n");

       for (int i=0;i<NBIN;i++){
           for(int j=i; j>=0; j--){
              // Print bin indices
              fprintf(OutFile,"%2d\t%2d\t",i,j);
              for(int k=0; k<=ORDER; k++) fprintf(OutFile,"%le\t",threepcf[i][j][k]);
              fprintf(OutFile,"\n");
            }
       }

       fflush(NULL);

       // Close open files
       fclose(OutFile);

       printf("\nOutput saved to %s\n",out_name);
     }


    inline void add_to_power(Multipoles *mult, Float wp) {
	// Now use all of the binned multipoles to compute the
	// spherical harmonics in all bins and then the cross-powers.
	// Need some scratch space:
	// This also applies the weight of the primary galaxy.
	Complex alm[NBIN][NLM];   // Apparently this initializes to zero

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

	    // We don't need to compute m<0, since it's just a complex conjugate.
	    // We will track the square of the coefficient elsewhere.
	    // Since only the square enters, the sign of the coefficient doesn't matter.
	    // Nor does a complex conjugate of the result.

// Bring in all of the tedious spherical harmonic code from another file.
#define CM(a,b,c) m[map[a][b][c]]
	    Complex *almbin = &(alm[i][0]);
#include "spherical_harmonics.cpp"

	 }

#define RealProduct(a,b) (a.real()*b.real()+a.imag()*b.imag())

  // COMPUTE 3PCF CONTRIBUTIONS

	for (int i=0; i<NBIN; i++) {
	    for (int j=0; j<=i; j++) {
		// Fill in a triangle of threepcf radial bins.
		// We want to compute the sum over m of Ylm[i] Ylm[j]^*
		// For m=0, that's just a_l0^2
		// For others, (a+bi)*(A-Bi) + (a-bi)*(A+Bi) = 2*(a*A+b*B)
		// We've put that factor of 2 in the weight3pcf[] array.

    // Note we use a different normalization convention to that of Slepian/Eisenstein 2015

    // NB: wp is the primary galaxy weight here

		for (int ell=0, n=0; ell<=ORDER; ell++) {
		    for (int mm=0; mm<=ell; mm++, n++) {
			// n counts the (ell,m)
			// Our definition is that the power is (-1)^ell / Sqrt(2ell+1) * Sum_m a_lm[a] * a_lm[b].conj()
			threepcf[i][j][ell] +=
			    wp*RealProduct(alm[i][n],alm[j][n])*weight3pcf[n]; //* 4.0 * M_PI / (2*ell+1.0);
		    }
		}
	     }
	}

  // COMPUTE 4PCF CONTRIBUTIONS

#ifdef FOURPCF
    printf("Add 4PCF computations here.\n\n");
#endif

	return;
    }

};  // end NPCF class

#endif
