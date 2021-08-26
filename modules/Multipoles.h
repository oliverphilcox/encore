#ifndef MULTIPOLES_H
#define MULTIPOLES_H

class Multipoles {
// This class will track the multipoles around a single primary for a given bin.
// The add() method will add one particle to the multipoles.
// However, for AVX, we have to gather 8 particles before we do the work.
// This is handled internally.  Call finish() at the end to make sure the last set is done.

  private:
    Float m[NMULT];	// Ambiguous as to whether this zeros out on construction
    int count;
    int nload;

#ifdef AVX
    // The assembly wants pointers to double4's.
  private:
    d4 *x1, *x2, *y1, *y2, *z1, *z2, *cx, *cy, *cz, *w1, *w2, *m4;
    Xdiff buf[8];	// We'll store up to 8 of these
#endif

  private:
    double empty[8];   // Just to try to keep the threads from working on similar memory

  public:

    Float *multipoles() { return m; }
    int ncount() { return count; }

    inline void reset_buffer() {
	nload = 0;
#ifdef AVX
 	double *m4p = (double *)m4;
	for (int kk=0; kk<4*NMULT; kk++) {
 	    m4p[kk] = 0.0;
 	}
#endif
	return;
    }

    inline void reset() {
	// Reset for a new primary particle
	for (int kk=0; kk<NMULT; kk++) m[kk] = 0.0;
	count = 0;
        reset_buffer();
	return;
    }

    inline void load_and_reset(Float *mptr, int *cptr) {
	// Reset for a new primary particle, but starting from an existing count
	for (int kk=0; kk<NMULT; kk++) m[kk] = mptr[kk];
	count = cptr[0];
        reset_buffer();
	return;
    }

    void save(Float *mptr, int *cptr) {
        // Write the m and count matrices out to these locations
	cptr[0] = count;
	for (int kk=0; kk<NMULT; kk++) mptr[kk] = m[kk];
	return;
    }

    Multipoles() {
#ifdef AVX
#define ALIGN 256
	// Allocate all of the aligned space for the AVX code
	int rv;
	rv = posix_memalign( (void **) &(x1), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(x2), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(y1), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(y2), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(z1), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(z2), ALIGN, 32 ); assert(rv==0);

	rv = posix_memalign( (void **) &(cx), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(cy), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(cz), ALIGN, 32 ); assert(rv==0);

	rv = posix_memalign( (void **) &(w1), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(w2), ALIGN, 32 ); assert(rv==0);
	rv = posix_memalign( (void **) &(m4), ALIGN, sizeof(double)*4*(NMULT+16)); assert(rv==0);

	// All of our multipoles are computed around the (0,0,0)
        cx->v[0] = 0.0; cx->v[1] = 0.0; cx->v[2] = 0.0; cx->v[3] = 0.0;
        cy->v[0] = 0.0; cy->v[1] = 0.0; cy->v[2] = 0.0; cy->v[3] = 0.0;
        cz->v[0] = 0.0; cz->v[1] = 0.0; cz->v[2] = 0.0; cz->v[3] = 0.0;
#endif
	empty[0] = 0.0;   // To avoid a warning
        reset();
	return;
    }
    ~Multipoles() {
#ifdef AVX
 	free(m4);
 	free(x1);
 	free(x2);
 	free(y1);
 	free(y2);
 	free(z1);
 	free(z2);
 	free(cx);
 	free(cy);
 	free(cz);
 	free(w1);
 	free(w2);
#endif
    }

    inline void add(Float xx, Float yy, Float zz, Float w) {
	// Add on the multipoles for a single secondary
	// No need to adjust nload; we're not putting the particle in a buffer
	count++;
	Float fi, fij, fijk;
	Float *mm = m;

        fi = w;
        for(int i=0;i<=ORDER;i++) {
            fij = fi;
            for(int j=0;j<=ORDER-i;j++) {
                fijk = fij;
                for(int k=0;k<=ORDER-i-j;k++) {
                    *mm += fijk;
		    fijk *= zz;
		    mm++;
                }
                fij *= yy;
            }
            fi *= xx;
        }
    }

    inline void finish() {
         // We never have any unfinished particles for non-AVX, so this is a null routine
#ifdef AVX
	if (ORDER==0) return;   // Didn't actually do AVX
	while (nload!=0) {
	    addAVX(0.0,0.0,0.0,0.0);   // Clear the buffer
	    count--;    // We don't want to count these as work.
	}
	// Sum up the 4-wide multipole buffers
	for (int i=0; i<NMULT; i++)
	    for (int n=0; n<4; n++) m[i] += m4[i].v[n];
#endif
	return;
    }

#ifdef AVX
    inline void addAVX(Float xx, Float yy, Float zz, Float w) {
	count++;
	// Load the buffer
	buf[nload].dx = xx;
	buf[nload].dy = yy;
	buf[nload].dz = zz;
	buf[nload].w = w;
	nload++;

	if (nload==8) {
	    // Full buffers; ready to compute the multipoles!
	    nload = 0;  // Reset the buffer

	    // For whatever reason, it is faster to do this transpose all at once
	    // than only as each particle is added.
            for (int n=0; n<4; n++) {
                x1->v[n] = buf[n].dx;
                x2->v[n] = buf[n+4].dx;

                y1->v[n] = buf[n].dy;
                y2->v[n] = buf[n+4].dy;

                z1->v[n] = buf[n].dz;
                z2->v[n] = buf[n+4].dz;

                w1->v[n] = buf[n].w;
                w2->v[n] = buf[n+4].w;
            }

	    // Call the AVX assembly code
	    (CMptr[ORDER-1])(x1,x2,y1,y2,z1,z2, cx,cy,cz, m4, w1,w2);
	}
    }
#endif

};   // END Multipoles class

#endif
