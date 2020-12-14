// grid_multipoles.cpp -- Daniel Eisenstein, started Oct 20, 2014.

/* DOCUMENTATION

Give the file name of the particles on the command line, as -in.
Will default to "sample.dat" if file not given.
Input file should be space-separated list of x,y,z,weight; one particle per line.
Lines starting with # are skipped as comments.
Positions will be rescaled by the -scale modifier.
Entering zero or negative sets this to the boxsize, so that
one can easily enter periodic particles in the unit cube.

Alternatively, using -ran <np> will throw np points randomly in the unit cube.

Variable boxsize (-box) then sets the scale of the cube.

The order of Y_lm being computed is set by the global definition ORDER.
This cannot exceed MAXORDER; the hardcoded Ylm's only go up to ell=10.
This must be set at compile time.

The number of radial bins is set by the global definition NBIN.
This must be set at compile time.

The binning is currently linear in radius, up to a maximum radius
'rmax', which is set by -rmax.

An important tuning parameter is ngrid, set by -ngrid.  This sets
the linear size of the grid that is used to speed the particle
search.  Overly large grid cells (small ngrid) will be inefficient
at rejecting pairs outside of the sphere of radius rmax.  Overly
small grid cells (large ngrid) incurs more overhead.  One would
like several grid cells per distance rmax, and one would like at
least dozens of particles per cell.

Running 'make' should compile two versions of the code.  One uses
AVX assembly to accelerate the multipole computation; the other doesn't.
The AVX appears to have more overhead, so for small ORDER, the non-AVX
code is actually faster.  The cross-over seems to be around ORDER==4.

The definition AVX should be set in the compile line by -DAVX and
causes the preprocessor to adjust the code appropriately.  Without
-DAVX set, the code should compile cleanly on a non-AVX computer.

On a Linux computer, altering the Makefile to
    CXX = icc -liomp5 -openmp
    CXXFLAGS = -O2 -Wall -DOPENMP
turns on multi-threading.  Without -DOPENMP, the code compiles just
fine without openMP installed (this is how I run on my Mac).  The code
still defines all of the extra buffers needed for multi-threading, but
then just uses the first one.

For large ORDER, the code shows good multi-threaded speed-up.  For small
order, however, the code currently only gets about a factor of 3-4 speed-up.
Apparently something about the pair finding is hitting a resource bottleneck.


As an advanced option, we can store and reload the multipoles per
particle and the binned pair counts.  The intention is to support
use cases in which one has a particle list with data particles
(weight >=0) and random particles (weight < 0).  One first runs the
code with only the data particles, saving the output.  One then
re-runs with a file that has some random particles postpended to
the input file.  By loading the stored file, the code will skip the
investigation of any pairs of positively weighted particles, thereby
re-using all of the DD work (and hence the DDD three-point).

The intention is that one might rerun the code many times with
different sets of random points.  Doing this will build up statistics
on the DR and RR counts.  By having each run use only n_R similar
to n_D, we avoid doing far too much work on RR as opposed to DR.
The optimum is to use 1.5--2 times more randoms than data in each run.

In this mode, only non-negative weighted primary particles have any
information stored, although they will search all secondary particles
when the stored file is created.  When re-loaded, only pairs of
non-negative primaries *and* secondaries are skipped.  So the initial
run should only have non-negative particles, or the book-keeping
will go askew.

The stored files are in a pure binary format; read the source code if you need
to parse it.  Note that this file isn't small: NBIN*NMULT*nparticle doubles!
For ORDER=4 and NBIN=10, that's 2800 bytes per particle.
For ORDER=10 and NBIN=10, 17600 bytes per particle.
Hopefully we can read it faster than we can recompute the DD!
BOSS-scaled DD appears to be about 5000 primaries/second on a 6-core machine,
so we need >1e8 Mbyte/sec to make this worthwhile.

*/



#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex>
#include <sys/time.h>
#include "threevector.hh"
#include "STimer.cc"

// For multi-threading:
#ifdef OPENMP
#include <omp.h>
#endif

// NBIN is the number of bins we'll sort the radii into.
#define NBIN 200

// ORDER is the order of the Ylm we'll compute.
// This must be <=MAXORDER, currently hard coded to 10.
#define ORDER 10

// MAXTHREAD is the maximum number of allowed threads.
// Big trouble if actual number exceeds this!
// No problem if actual number is smaller.
#define MAXTHREAD 40

typedef unsigned long long int uint64;

// Could swap between single and double precision here.
// Only double precision has been tested.
// Note that the AVX multipole code is always double precision.
typedef double Float;
typedef double3 Float3;
typedef std::complex<double> Complex;

// We need a vector floor3 function
Float3 floor3(float3 p) {
    return Float3(floor(p.x), floor(p.y), floor(p.z));
}

#define PAGE 4096     // To force some memory alignment.

// =================== Particles ====================
// This is the info about each particle that we load in and store in the Grid.

class Particle {
  public:
    Float3 pos;
    Float w;  // The weight for each particle
};

// ====================  The Cell and Grid classes ==================

/* The Grid class holds a new copy of the particles.
These are sorted into cells, and all positions are referenced to the
cell center.  That way, we can handle periodic wrapping transparently,
simply by the cell indexing.

For simplicity, we opt to flatten the index of the cells into a 1-d number.
For example, this makes multi-threading over cells simpler.
*/

class Cell {
  public:
    int start;	// The starting index of the particle list
    int np;
};

class Grid {
  public:
    Float boxsize;   // Size of the periodic volume
    int nside, ncells;       // Grid size (per linear and per volume)
    Cell *c;		// The list of cells
    Float cellsize;   // Size of one cell
    Particle *p;	// Pointer to the list of particles
    int np;		// Number of particles
    int np_pos;		// Number of particles
    int *pid;		// The original ordering
    Float sumw_pos, sumw_neg; // Summing the weights

    int wrap_cell(integer3 cell) {
        // Return the 1-d cell number, after wrapping
	// We apply a very large bias, so that we're
	// guaranteed to wrap any reasonable input.
	int cx = (cell.x+ncells)%nside;
	int cy = (cell.y+ncells)%nside;
	int cz = (cell.z+ncells)%nside;
	// return (cx*nside+cy)*nside+cz;
	int answer = (cx*nside+cy)*nside+cz;
	assert(answer<ncells&&answer>=0);
	/* printf("Cell: %d %d %d -> %d %d %d -> %d\n",
	    cell.x, cell.y, cell.z, cx, cy, cz, answer); */
	return answer;
    }

    integer3 cell_id_from_1d(int n) {
	// Undo 1d back to 3-d indexing
        assert(n>=0&&n<ncells);
	// printf("Cell ID: %d ", n);
	integer3 cid;
	cid.z = n%nside;
	n = n/nside;
	cid.y = n%nside;
	cid.x = n/nside;
	// printf("-> %d %d %d\n", cid.x, cid.y, cid.z);
	return cid;
    }

    int pos_to_cell(Float3 pos) {
        // Return the 1-d cell number for this position, properly wrapped
	// We assume the first cell is centered at cellsize/2.0
	// return wrap_cell( floor3(pos/cellsize+Float3(0.5,0.5,0.5)));
	return wrap_cell( floor3(pos/cellsize));
    }

    Float3 cell_centered_pos(Float3 pos) {
        // Subtract off the cell center from the given position.
	// This is safe for positions not in the primary box.
	return pos-cellsize*(floor3(pos/cellsize)+Float3(0.5,0.5,0.5));
    }

    Float3 cell_sep(integer3 sep) {
	// Return the position difference corresponding to a cell separation
        return cellsize*sep;
    }

    ~Grid() {
	// The destructor
        free(p);
	free(pid);
	free(c);
	return;
    }

    Grid(Particle *input, int _np, Float _boxsize, int _nside) {
	// The constructor: the input set of particles is copied into a
	// new list, which is ordered by cell.
	// After this, Grid is self-sufficient; one could discard *input
        boxsize = _boxsize;
	nside = _nside;
	assert(nside<1025);   // Can't guarantee won't spill int32 if bigger
	ncells = nside*nside*nside;
	np = _np;
	np_pos = 0;
	assert(boxsize>0&&nside>0&&np>=0);
	cellsize = boxsize/nside;

	p = (Particle *)malloc(sizeof(Particle)*np);
	pid = (int *)malloc(sizeof(int)*np);
	printf("# Allocating %6.3f MB of particles\n", (sizeof(Particle)+sizeof(int))*np/1024.0/1024.0);
	c = (Cell *)malloc(sizeof(Cell)*ncells);
	printf("# Allocating %6.3f MB of cells\n", (sizeof(Cell))*ncells/1024.0/1024.0);

	// Now we want to copy the particles, but do so into grid order.

	// First, figure out the cell for each particle
	int *cell = (int *)malloc(sizeof(int)*np);
	for (int j=0; j<np; j++) cell[j] = pos_to_cell(input[j].pos);

	// Histogram the number of particles in each cell
	int *incell = (int *)malloc(sizeof(int)*ncells);
	for (int j=0; j<ncells; j++) incell[j] = 0.0;
	for (int j=0; j<np; j++) incell[cell[j]]++;

	// Count the number of positively weighted particles
	sumw_pos = sumw_neg = 0.0;
	for (int j=0; j<np; j++)
	    if (input[j].w>=0) {
	    	np_pos++;
		sumw_pos += input[j].w;
	    } else {
		sumw_neg += input[j].w;
	    }

	// Cumulate the histogram, so we know where to start each cell
	for (int j=0, tot=0; j<ncells; tot+=incell[j], j++) {
	    c[j].start = tot;
	    c[j].np = 0;  // We'll count these as we add the particles
	}

	// Copy the particles into the cell-ordered list
	for (int j=0; j<np; j++) {
	    Cell *thiscell = c+cell[j];
	    int index = thiscell->start+thiscell->np;
	    p[index] = input[j];
	    p[index].pos = cell_centered_pos(input[j].pos);
	    	// Switch to cell-centered positions
	    pid[index] = j;	 // Storing the original index

	    // Diagnostics:
	    /*
	    integer3 cid = cell_id_from_1d(cell[j]);
	    printf("P->C: %d %7.4f %7.4f %7.4f -> %d (%d %d %d) %7.4f %7.4f %7.4f %d\n",
		j, input[j].pos.x, input[j].pos.y, input[j].pos.z,
		cell[j], cid.x, cid.y, cid.z,
		p[index].pos.x, p[index].pos.y, p[index].pos.z, index);
	    */

	    thiscell->np += 1;
	}

	// Checking that all is well.
	int tot = 0;
	for (int j=0; j<ncells; j++) {
	    assert(c[j].start == tot);
	    assert(c[j].np == incell[j]);
	    tot += c[j].np;
	}
	free(incell);
	assert(tot == np);

	free(cell);
	return;
    }

};   // End Grid class



// ========================== Accumulate the pair counts ================

class Pairs {
  private:
    double *xi0, *xi2;

  private:
    double empty[8];   // Just to try to keep the threads from working on similar memory

  public:
    Pairs() {
	// Initialize the binning
	posix_memalign((void **) &xi0, PAGE, sizeof(double)*NBIN);
	posix_memalign((void **) &xi2, PAGE, sizeof(double)*NBIN);
	for (int j=0; j<NBIN; j++) {
	    xi0[j] = 0;
	    xi2[j] = 0;
	}
	empty[0] = 0.0;   // To avoid a warning
    }
    ~Pairs() {
        free(xi0);
	free(xi2);
    }

    inline void load(Float *xi0ptr, Float *xi2ptr) {
	for (int j=0; j<NBIN; j++) {
	    xi0[j] = xi0ptr[j];
	    xi2[j] = xi2ptr[j];
	}
    }
    inline void save(Float *xi0ptr, Float *xi2ptr) {
	for (int j=0; j<NBIN; j++) {
	    xi0ptr[j] = xi0[j];
	    xi2ptr[j] = xi2[j];
	}
    }

    inline void add(int b, Float dz, Float w) {
	// Add up the weighted pair for the monopole and quadrupole correlation function
        xi0[b] += w;
        xi2[b] += w*(3.0*dz*dz-1)*0.5;
    }

    void sum_power(Pairs *p) {
	// Just add up all of the threaded pairs into the zeroth element
	for (int i=0; i<NBIN; i++) {
        xi0[i] += p->xi0[i];
	    xi2[i] += p->xi2[i];
	}
    }

    void report_pairs() {
    for (int j=0; j<NBIN; j++) {
	    printf("Pairs %2d %9.0f %9.0f\n",
			j, xi0[j], xi2[j]);
	}
    }
};

Pairs pairs[MAXTHREAD];

// ====================  Setting up the multipoles ==================

// The total number of Cartesian multipoles, satisfying a+b+c<=ORDER
#define NMULT ((ORDER+1)*(ORDER+2)*(ORDER+3)/6)
// We adopt a convention in which we loop over ell and then m=0..ell
#define NLM ((ORDER+1)*(ORDER+2)/2)

#ifdef AVX
#include "externalmultipoles.h"
// typedef struct { double v[4]; } d4;

// An array of pointers to all of the AVX assembly functions
void (*CMptr[16])( d4 *ip1x, d4 *ip2x, d4 *ip1y, d4 *ip2y, d4 *ip1z, d4 *ip2z,
                   d4 *cx, d4 *cy, d4 *cz, d4 *globalM,
                   d4 *mass1, d4 *mass2) = {
     MultipoleKernel1,  MultipoleKernel2,  MultipoleKernel3,  MultipoleKernel4,
     MultipoleKernel5,  MultipoleKernel6,  MultipoleKernel7,  MultipoleKernel8,
     MultipoleKernel9,  MultipoleKernel10, MultipoleKernel11, MultipoleKernel12,
     MultipoleKernel13, MultipoleKernel14, MultipoleKernel15, MultipoleKernel16
};

#endif

// Here's a simple structure for our normalized differences of the positions
typedef struct Xdiff {
    Float dx, dy, dz, w;
} Xdiff;

class Multipoles {
// This class will track the multipoles around a single primary for a given bin.
// The add() method will add one particle to the multipoles.
// However, for AVX, we have to gather 8 particles before we do the work.
// This is handled internally.  Call finish() at the end to make sure the last set is done.

  private:
    Float m[NMULT];	// Ambiguous as to whether this zeros out on construction
    uint64 count;
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
    uint64 ncount() { return count; }

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

    inline void load_and_reset(Float *mptr, uint64 *cptr) {
	// Reset for a new primary particle, but starting from an existing count
	for (int kk=0; kk<NMULT; kk++) m[kk] = mptr[kk];
	count = cptr[0];
        reset_buffer();
	return;
    }

    void save(Float *mptr, uint64 *cptr) {
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




// ========================== Storing Multipoles ==========================

/* We will install the option to store and load the multipoles for positively
weighted particles.  The intended use is that one has a set of data particles
at the front of the input particle file, followed by a set of random particles
with negative weight.  We don't want to repeat the DD counting if we change the
randoms.  This can be accomplished by storing the DD multipoles and then skipping
positively weighted pairs in subsequent passes.  We simply need to overwrite the
m[] array in the multipoles before starting the counts.

For each particle, we use the pid to index the result.

This formalism requires that all of the w>=0 particles are at the top of the input
file, followed by w<0 particles.  And the error checking of this assumption is poor!
*/

class StoreMultipoles {
  private:
    Float *m[NBIN];    // Each is a flattened array [np_pos][NMULT]
    uint64 *count;    // Flattened array [np_pos][NBIN]
    int np_pos;
  public:
    Float *xi0, *xi2;   // Flattened arrays [NBIN]

    StoreMultipoles(int _np_pos) {
	np_pos = _np_pos;
	for (int b=0;b<NBIN;b++)
	    m[b] = (Float *)malloc(sizeof(Float)*np_pos*NMULT);
        count = (uint64 *)malloc(sizeof(uint64)*np_pos*NBIN);
        xi0 = (Float *)malloc(sizeof(Float)*NBIN);
        xi2 = (Float *)malloc(sizeof(Float)*NBIN);
	printf("# Allocating %6.3f MB\n",
		(sizeof(Float)*np_pos*NMULT*NBIN+sizeof(uint64)*np_pos*NBIN)/1024.0/1024.0);
    }
    ~StoreMultipoles() {
	for (int b=0;b<NBIN;b++) free(m[b]);
	free(count);
	free(xi0);
	free(xi2);
    }

    void save(char fname[]) {
	// Dump all of the data to a file
	FILE *fp = fopen(fname, "wb");
	assert(fp);
	int tmp[4];
	tmp[0] = np_pos; tmp[1] = NBIN; tmp[2] = NMULT; tmp[3] = ORDER;
	fwrite((void *)&tmp, sizeof(int), 4, fp);
	fwrite((void *)xi0, sizeof(Float), NBIN, fp);
	fwrite((void *)xi2, sizeof(Float), NBIN, fp);
	for (int b=0;b<NBIN;b++)
	    fwrite((void *)m[b], sizeof(Float), np_pos*NMULT, fp);
	fwrite((void *)count, sizeof(uint64),  np_pos*NBIN, fp);
	fclose(fp);
	printf("# Saving counts of %d objects to file: %s\n", np_pos, fname);
        return;
    }

    void load(char fname[]) {
        // Restore data from a file
	FILE *fp = fopen(fname, "rb");
	assert(fp);
	size_t obj;
	int tmp[4];
	obj = fread((void *)&tmp, sizeof(int), 4, fp);
	assert(obj == 4);
	assert(tmp[0] == np_pos);
	assert(tmp[1] == NBIN);
	assert(tmp[2] == NMULT);
	assert(tmp[3] == ORDER);
	obj = fread((void *)xi0, sizeof(Float), NBIN, fp);
	assert(obj == NBIN);
	obj = fread((void *)xi2, sizeof(Float), NBIN, fp);
	assert(obj == NBIN);
	for (int b=0;b<NBIN;b++) {
	    obj = fread((void *)m[b], sizeof(Float), np_pos*NMULT, fp);
	    assert(obj == (size_t)np_pos*NMULT);
	}
	obj = fread((void *)count, sizeof(uint64),  np_pos*NBIN, fp);
	assert(obj == (size_t)np_pos*NBIN);
	fclose(fp);
	printf("# Successfully loaded counts of %d objects from file: %s\n", np_pos, fname);
        return;
    }

    Float *fetchM(int pid, int bin) {
	// Return a pointer to the NMULT array for this particle and bin
	return m[bin]+pid*NMULT;
    }

    uint64 *fetchC(int pid, int bin) {
	// Return a pointer to the count variable for this particle and bin
	return count+pid*NBIN+bin;
    }
};

StoreMultipoles *smload, *smsave;

// ========================== Here are all of the cross-powers ============

#define MAXORDER 10
#define NLM_MAX ((MAXORDER+1)*(MAXORDER+2)/2)
// Some global constants for the a_lm normalizations.
// From: http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
// Including an extra factor of 2 in all m!=0 cases.

// All factors are of the form a*sqrt(b/pi), so let's use that:
#define YNORM(a,b) (2.0*(1.0*a)*(1.0*a)*(1.0*b)/M_PI)
// This includes the factor of 2, so divide the m=0 by 2.

static Float almnorm[NLM_MAX] = {
    YNORM(1/2,1)/2.0,

    YNORM(1/2,3)/2.0,
    YNORM(1/2,3/2),

    YNORM(1/4,5)/2.0,
    YNORM(1/2,15/2),
    YNORM(1/4,15/2),

    YNORM(1/4,7)/2.0,
    YNORM(1/8,21),
    YNORM(1/4,105/2),
    YNORM(1/8,35),

    YNORM(3/16,1)/2.0,
    YNORM(3/8,5),
    YNORM(3/8,5/2),
    YNORM(3/8, 35),
    YNORM(3/16, 35/2),

    YNORM(1/16, 11)/2.0,
    YNORM(1/16, 165/2),
    YNORM(1/8, 1155/2),
    YNORM(1/32, 385),
    YNORM(3/16, 385/2),
    YNORM(3/32, 77),

    YNORM(1/32, 13)/2.0,
    YNORM(1/16, 273/2),
    YNORM(1/64, 1365),
    YNORM(1/32, 1365),
    YNORM(3/32, 91/2),
    YNORM(3/32, 1001),
    YNORM(1/64, 3003),

    YNORM(1/32, 15)/2.0,
    YNORM(1/64, 105/2),
    YNORM(3/64, 35),
    YNORM(3/64, 35/2),
    YNORM(3/32, 385/2),
    YNORM(3/64, 385/2),
    YNORM(3/64, 5005),
    YNORM(3/64, 715/2),

    YNORM(1/256, 17)/2.0,
    YNORM(3/64, 17/2),
    YNORM(3/128, 595),
    YNORM(1/64, 19635/2),
    YNORM(3/128, 1309/2),
    YNORM(3/64, 17017/2),
    YNORM(1/128, 7293),
    YNORM(3/64, 12155/2),
    YNORM(3/256, 12155/2),

    YNORM(1/256, 19)/2.0,
    YNORM(3/256, 95/2),
    YNORM(3/128, 1045),
    YNORM(1/256, 21945),
    YNORM(3/128, 95095/2),
    YNORM(3/256, 2717),
    YNORM(1/128, 40755),
    YNORM(3/512, 13585),
    YNORM(3/256, 230945/2),
    YNORM(1/512, 230945),

    YNORM(1/512, 21)/2.0,
    YNORM(1/256, 1155/2),
    YNORM(3/512, 385/2),
    YNORM(3/256, 5005),
    YNORM(3/256, 5005/2),
    YNORM(3/256, 1001),
    YNORM(3/1024, 5005),
    YNORM(3/512, 85085),
    YNORM(1/512, 255255/2),
    YNORM(1/512, 4849845),
    YNORM(1/1024, 969969)
};


class CPower {
// This should accumulate the C_ell power, for all combination of bins.
  public:
    uint64 bincounts[NBIN];
    Float binweight[NBIN];
    Float cpower[NBIN][NBIN][ORDER+1];
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
		    cpower[i][j][k] = 0.0;
		}
	    }
	}
	return;
    }

    CPower() {
	make_map();
        reset();
	return;
    }
    ~CPower() {
    }

    void sum_power(CPower *c) {
	// Just add up all of the threaded power into the zeroth element
	for (int i=0; i<NBIN; i++) {
	    bincounts[i] += c->bincounts[i];
	    binweight[i] += c->binweight[i];
	    for (int j=0; j<NBIN; j++) {
		for (int k=0; k<=ORDER; k++) {
		    cpower[i][j][k] += c->cpower[i][j][k];
		}
	    }
	}
    }

    void report_power() {
	for (int i=0; i<NBIN; i++) {
	    for (int j=i; j>=0; j--) {
		if (j==i) printf("Multipole Power %2d %9lld %9.0f\n",
				j, bincounts[j], binweight[j]);
		//if (j==i) {
		//    printf("# Bin auto-power omitted due to uncorrected noise bias\n");
		//    continue;
		//}
		printf("%2d %2d", i, j);
		printf(" %13.6e", cpower[i][j][0]);
		for (int k=1; k<=ORDER; k++) {
		    printf(" %13.6e", cpower[i][j][k]/cpower[i][j][0]);
		}
	    printf("\n");
	    }
	}
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

	// Next we'll accumulate powers in all cross-bins.
	for (int i=0; i<NBIN; i++) {
	    for (int j=0; j<=i; j++) {
		// Fill in a triangle of cpower.
		// We want to compute the sum over m of Ylm[i] Ylm[j]^*
		// For m=0, that's just a_l0^2
		// For others, (a+bi)*(A-Bi) + (a-bi)*(A+Bi) = 2*(a*A+b*B)
		// We've put that factor of 2 in the almnorm[] array.

		for (int ell=0, n=0; ell<=ORDER; ell++) {
		    for (int mm=0; mm<=ell; mm++, n++) {
			// n counts the (ell,m)
			// Our definition is that the power is 4*pi/(2*l+1)
			// times alm alm*.
			cpower[i][j][ell] +=
			    wp*RealProduct(alm[i][n],alm[j][n])*almnorm[n]
			    * 4.0 * M_PI / (2*ell+1.0);
		    }
		}
	     }
	}
	return;
    }

};  // end CPower class


CPower cpower[MAXTHREAD];

void zero_power() {
    for (int t=0; t<MAXTHREAD; t++) cpower[t].reset();
}

void sum_power() {
    // Just add up all of the threaded power into the zeroth element
    for (int t=0; t<MAXTHREAD; t++)
	printf("# Bin 0 counter for thread %2d: %9lld\n", t, cpower[t].bincounts[0]);
    for (int t=1; t<MAXTHREAD; t++)
        cpower[0].sum_power(cpower+t);
    for (int t=1; t<MAXTHREAD; t++)
        pairs[0].sum_power(pairs+t);
    return;
}


// ====================  Computing the pairs ==================

void compute_multipoles(Grid *grid, Float rmax) {
    int maxsep = ceil(rmax/grid->cellsize);   // How far we must search
    int ne;
    Float rmax2 = rmax*rmax;
    Float rmin2 = rmax2*1e-12;    // Just an underflow guard
    uint64 cnt = 0;

    Multipoles *mlist = new Multipoles[MAXTHREAD*NBIN];  // Set up all of this space

    // Easy to multi-thread this top loop!
    // But some cells have trivial amounts of work, so we will first make a list of the work.
    // Including the empty cells appears to fool the dynamic thread allocation sometimes.
    int non_empty_cell[grid->ncells], non_empty=0;
    for (int n=0; n<grid->ncells; n++) {
        if (grid->c[n].np>0) non_empty_cell[non_empty++] = n;
    }
    printf("# Found %d non-empty cells.\n", non_empty);

    // We're going to loop only over the non-empty cells.
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic,8) reduction(+:cnt)
#endif
    for (ne=0; ne<non_empty; ne++) {
	int n = non_empty_cell[ne];  // Fetch the cell number
	// Decide which thread we are in
#ifdef OPENMP
	int thread = omp_get_thread_num();
        assert(omp_get_num_threads()<=MAXTHREAD);
        if (ne==0) printf("# Running on %d threads.\n", omp_get_num_threads());
#else
	int thread = 0;
        if (ne==0) printf("# Running single threaded.\n");
#endif
    	// Loop over primary cells.
	Cell primary = grid->c[n];
	integer3 prim_id = grid->cell_id_from_1d(n);

	Multipoles *mult = mlist+thread*NBIN;   // Workspace for this thread

	// continue; // To skip all of the list-building and summations.
		// Everything else takes negligible time

	// Now we need to loop over all primary particles in this cell
	for (int j = primary.start; j<primary.start+primary.np; j++) {
	    int mloaded = 0;
	    if (smload && grid->p[j].w>=0) {
		// Start the multipoles from the input values
		// ONLY if the primary particle has weight>0
		int pid = grid->pid[j];
		for (int b=0; b<NBIN; b++)
		    mult[b].load_and_reset(smload->fetchM(pid,b), smload->fetchC(pid,b));
		mloaded = 1;    // We'll use this to skip some pairs later.
	    } else {
		for (int b=0; b<NBIN; b++) mult[b].reset();   // Zero out the multipoles
	    }

	    Float primary_w = grid->p[j].w;

	    // Then loop over secondaries, cell-by-cell
	    integer3 delta;
	    for (delta.x = -maxsep; delta.x <= maxsep; delta.x++)
	    for (delta.y = -maxsep; delta.y <= maxsep; delta.y++)
	    for (delta.z = -maxsep; delta.z <= maxsep; delta.z++) {
		const int samecell = (delta.x==0&&delta.y==0&&delta.z==0)?1:0;
		Cell sec = grid->c[grid->wrap_cell(prim_id+delta)];
	        Float3 ppos = grid->p[j].pos - grid->cell_sep(delta);
		// This is the position of the particle as viewed from the
		// secondary cell.
		// Now loop over the particles in this secondary cell
		for (int k = sec.start; k<sec.start+sec.np; k++) {
		    // Now we're considering these two particles!
		    if (samecell&&j==k) continue;   // Exclude self-count
		    if (mloaded && grid->p[k].w>=0) continue;
		    	// This particle has already been included in the file we loaded.
		    Float3 dx = grid->p[k].pos - ppos;
		    Float norm2 = dx.norm2();
		    if (norm2<rmax2 && norm2>rmin2) cnt++; else continue;

		    // Now what do we want to do with the pair?
		    norm2 = sqrt(norm2);  // Now just radius
		    // Find the radial bin
		    int bin = floor(norm2/rmax*NBIN);
		    /*
		    printf("%d %d  %d %f (%f %f %f)\n",
		    	grid->pid[j], grid->pid[k], bin,
			norm2, dx.x, dx.y, dx.z);
		    */

		    dx = dx/norm2;

		    // continue;   // Skip pairs and multipoles

		    // Accumulate the 2-pt correlation function
		    // We include the weight for each pair
            pairs[thread].add(bin, dx.z, grid->p[k].w*primary_w);

		    //continue;   // Skip the multipole creation

		    // Accumulate the multipoles
#ifdef AVX 	    // AVX only available for ORDER>=1
		    if (ORDER) mult[bin].addAVX(dx.x, dx.y, dx.z, grid->p[k].w);
			    else  mult[bin].add(dx.x, dx.y, dx.z, grid->p[k].w);
#else
		    mult[bin].add(dx.x, dx.y, dx.z, grid->p[k].w);
#endif
		} // Done with this secondary particle
	    } // Done with this secondary cell
	    for (int b=0; b<NBIN; b++) mult[b].finish();   // Finish the multipoles

	    if (smsave && grid->p[j].w>=0) {
	        // We're saving multipoles, and this particle has positive weight.
		int pid = grid->pid[j];
		for (int b=0; b<NBIN; b++) mult[b].save(smsave->fetchM(pid,b), smsave->fetchC(pid,b));
	    }

	    // Now add these multipoles into the cross-powers
	    // This step takes very little time.
	    // continue;     // Skip the power summation
	    cpower[thread].add_to_power(mult, primary_w);

	} // Done with this primary particle
    } // Done with this primary cell, end of omp pragma

    printf("# We counted  %lld pairs within %f.\n", cnt, rmax);
    printf("# Average of %f pairs per primary particle.\n",
    		(Float)cnt/grid->np);
    float x = rmax/grid->boxsize;
    float expected = grid->np * (4*M_PI/3.0)*grid->np*x*x*x;
    printf("# We expected %1.0f pairs, off by %f.\n", expected, cnt/expected);
    delete[] mlist;
    return;
}




// ====================  The Driver ===========================

Particle *make_particles(Float boxsize, int np) {
    // Make np random particles
    srand48(1);      // For reproducibility
    Particle *p = (Particle *)malloc(sizeof(Particle)*np);
    for (int j=0; j<np; j++) {
        p[j].pos.x = drand48()*boxsize;
        p[j].pos.y = drand48()*boxsize;
        p[j].pos.z = drand48()*boxsize;
        p[j].w = 1.0;     // For all positive weights
        //p[j].w = (j%2==0)?1.0:-1.0;   // To get an equal number of positive and negative weights
    }
    printf("# Done making %d random particles, periodically distributed.\n", np);
    return p;
}

Particle *read_particles(Float rescale, int *np, const char *filename) {
    // This will read particles from a file, space-separated x,y,z,w
    // Particle positions will be rescaled by the variable 'rescale'.
    // For example, if rescale==boxsize, then inputing the unit cube will cover the periodic volume
    char line[1000];
    int j=0, n=0;
    FILE *fp;
    double tmp[4];
    fp = fopen(filename, "r");
    if (fp==NULL) {
        fprintf(stderr,"File %s not found\n", filename); abort();
    }
    while (fgets(line,1000,fp)!=NULL) {
        if (line[0]=='#') continue;
	n++;
    }
    rewind(fp);
    *np = n;
    Particle *p = (Particle *)malloc(sizeof(Particle)*n);
    printf("# Found %d particles from %s\n", n, filename);
    printf("# Rescaling input positions by factor %f\n", rescale);
    while (fgets(line,1000,fp)!=NULL) {
        if (line[0]=='#') continue;
	if (sscanf(line, "%lf %lf %lf %lf", tmp, tmp+1, tmp+2, tmp+3)!=4) {
	    fprintf(stderr,"Particle %d has bad format\n", j);
	    abort();
	}
	if (tmp[3]==0.0) { *np -= 1; continue; }
		// Skip any objects with no weight; reduce the particle count
        p[j].pos.x = tmp[0]*rescale;
        p[j].pos.y = tmp[1]*rescale;
        p[j].pos.z = tmp[2]*rescale;
        p[j].w = tmp[3];
	j++;
    }
    fclose(fp);
    printf("# Done reading the particles\n");
    return p;
}

void check_bounding_box(Particle *p, int np, Float boxsize, Float rmax) {
    // Check that the bounding box is reasonable
    Float3 pmin, pmax;
    pmin.x = pmin.y = pmin.z = 1e30;
    pmax.x = pmax.y = pmax.z = -1e30;
    for (int j=0; j<np; j++) {
        pmin.x = fmin(pmin.x, p[j].pos.x);
        pmin.y = fmin(pmin.y, p[j].pos.y);
        pmin.z = fmin(pmin.z, p[j].pos.z);
        pmax.x = fmax(pmax.x, p[j].pos.x);
        pmax.y = fmax(pmax.y, p[j].pos.y);
        pmax.z = fmax(pmax.z, p[j].pos.z);

	// if (p[j].pos.x>max.x) max.x = p[j].pos.x;
	// if (p[j].pos.y>max.y) max.y = p[j].pos.y;
	// if (p[j].pos.z>max.z) max.z = p[j].pos.z;
	// if (p[j].pos.x<min.x) min.x = p[j].pos.x;
	// if (p[j].pos.y<min.y) min.y = p[j].pos.y;
	// if (p[j].pos.z<min.z) min.z = p[j].pos.z;
    }
    printf("# Range of x positions are %6.2f to %6.2f\n", pmin.x, pmax.x);
    printf("# Range of y positions are %6.2f to %6.2f\n", pmin.y, pmax.y);
    printf("# Range of z positions are %6.2f to %6.2f\n", pmin.z, pmax.z);
    Float3 prange = pmax-pmin;
    Float         biggest = prange.x;
    biggest = fmax(biggest, prange.y);
    biggest = fmax(biggest, prange.z);
    printf("# Biggest range is %6.2f\n", biggest);
    if (biggest>boxsize*1.001)
	printf("#\n# WARNING: particles will overlap on period wrapping!\n#\n");
    if (biggest+rmax<boxsize*0.6)
	printf("#\n# WARNING: box periodicity seems too generous, will hurt grid efficiency!\n#\n");

    if (prange.x>0.99*biggest && prange.y>0.99*biggest && prange.z>0.99*biggest) {
        // Probably using a cube of inputs, intended for a periodic box
	if (biggest<0.99*boxsize)
	    printf("#\n# WARNING: cubic input detected, but smaller than periodicity!\n#\n");
    } else {
        // Probably a non-periodic input
	if (biggest+rmax > boxsize)
	    printf("#\n# WARNING: non-cubic input detected, but could overlap periodically!\n#\n");
    }
    return;
}

void invert_weights(Particle *p, int np) {
    for (int j=0; j<np; j++) p[j].w *= -1.0;
    printf("# Multiplying all weights by -1\n");
}

void balance_weights(Particle *p, int np) {
    Float sumpos = 0.0, sumneg = 0.0;
    for (int j=0; j<np; j++)
	if (p[j].w>=0.0) sumpos += p[j].w;
	    else sumneg += p[j].w;
    if (sumneg==0.0 || sumpos==0.0) {
	fprintf(stderr,"Asked to rebalance weights, but there are not both positive and negative weights\n");
	abort();
    }
    Float rescale = sumpos/(-sumneg);
    printf("# Rescaling negative weights by %f\n", rescale);
    for (int j=0; j<np; j++)
	if (p[j].w<0.0) p[j].w *= rescale;
    return;
}

// ================================ main() =============================

void usage() {
    fprintf(stderr, "\nUsage for grid_multipoles/grid_multipolesAVX:\n");
    fprintf(stderr, "   -box <boxsize> : The periodic size of the computational domain.  Default 400.\n");
    fprintf(stderr, "   -scale <rescale>: How much to dilate the input positions by.  Default 0.\n");
    fprintf(stderr, "             Zero or negative value causes =boxsize, rescaling unit cube to full periodicity\n");
    fprintf(stderr, "   -rmax <rmax>: The maximum radius of the largest pair bin.  Default 200.\n");
    fprintf(stderr, "   -nside <nside>: The grid size for accelerating the pair count.  Default 8.\n");
    fprintf(stderr, "             Recommend having several grid cells per rmax.\n");
    fprintf(stderr, "   -in <file>: The input file (space-separated x,y,z,w).  Default sample.dat.\n");
    fprintf(stderr, "   -ran <np>: Ignore any file and use np random perioidic points instead.\n");
    fprintf(stderr, "   -def: This allows one to accept the defaults without giving other entries.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Two other important parameters can only be set during compilations:\n");
    fprintf(stderr, "   ORDER: The multipole order being computed.\n");
    fprintf(stderr, "   NBIN:  The number of radial bins.\n");
    fprintf(stderr, "Similarly, the radial bin spacing (currently linear) is hard-coded.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "For advanced use, there is an option store the multipoles of positively weighted primary particles.\n");
    fprintf(stderr, "    -save <filename>: Triggers option to store the multipoles.\n");
    fprintf(stderr, "The file can then be reloaded on subsequent runes\n");
    fprintf(stderr, "    -load <filename>: Triggers option to load the multipoles\n");
    fprintf(stderr, "The intention is to allow re-use of DD counts while changing the DR and RR counts.\n");
    fprintf(stderr, "    -balance: Rescale the negative weights so that the total weight is zero.\n");
    fprintf(stderr, "    -invert: Multiply all the weights by -1.\n");


    exit(1);
    return;
}

int main(int argc, char *argv[]) {
    // Important variables to set!  Here are the defaults:
    Float boxsize = 1;
        // The periodicity of the position-space cube.
    Float rescale = 0.0;   // If left zero or negative, set rescale=boxsize
    	// The particles will be read from the unit cube, but then scaled by boxsize.
    Float rmax = 0.05;
    	// The maximum radius of the largest bin.
    int nside = 20;
	// The grid size, which should be tuned to match boxsize and rmax.
        // Don't forget to adjust this if changing boxsize!
    int make_random = 0;
    	// If set, we'll just throw random periodic points instead of reading the file
    int np = -1;   // Will be number of particles in a random distribution,
    	// but gets overwritten if reading from a file.
    int qbalance = 0, qinvert = 0;
    const char default_fname[] = "sample.dat";
    char *fname = NULL;
    char *savename = NULL;
    char *loadname = NULL;

    STimer TotalTime, Prologue, Epilogue, MultipoleTime, IOTime;

    TotalTime.Start();
    Prologue.Start();
    if (argc==1) usage();
    int i=1;
    while (i<argc) {
	     if (!strcmp(argv[i],"-boxsize")||!strcmp(argv[i],"-box")) boxsize = atof(argv[++i]);
	else if (!strcmp(argv[i],"-rescale")||!strcmp(argv[i],"-scale")) rescale = atof(argv[++i]);
	else if (!strcmp(argv[i],"-rmax")||!strcmp(argv[i],"-max")) rmax = atof(argv[++i]);
	else if (!strcmp(argv[i],"-nside")||!strcmp(argv[i],"-ngrid")||!strcmp(argv[i],"-grid")) nside = atoi(argv[++i]);
	else if (!strcmp(argv[i],"-in")) fname = argv[++i];
	else if (!strcmp(argv[i],"-save")||!strcmp(argv[i],"-store")) savename = argv[++i];
	else if (!strcmp(argv[i],"-load")) loadname = argv[++i];
	else if (!strcmp(argv[i],"-balance")) qbalance = 1;
	else if (!strcmp(argv[i],"-invert")) qinvert = 1;
	else if (!strcmp(argv[i],"-ran")||!strcmp(argv[i],"-np")) {
		double tmp;
		if (sscanf(argv[++i],"%lf", &tmp)!=1) {
		    fprintf(stderr, "Failed to read number in %s %s\n",
		    	argv[i-1], argv[i]);
		    usage();
		}
		np = tmp;
		make_random=1;
	    }
	else if (!strcmp(argv[i],"-def")||!strcmp(argv[i],"-default")) { fname = NULL; }
	else {
	    fprintf(stderr, "Don't recognize %s\n", argv[i]);
	    usage();
	}
	i++;
    }
    assert(i==argc);  // For example, we might have omitted the last argument, causing disaster.

    assert(boxsize>0.0);
    assert(rmax>0.0);
    assert(nside>0);
    assert(nside<300);   // Legal, but rather unlikely that we should use something this big!
    if (rescale<=0.0) rescale = boxsize;   // This would allow a unit cube to fill the periodic volume
    if (fname==NULL) fname = (char *) default_fname;   // No name was given

    // Output for posterity
    printf("Box Size = %6.1g\n", boxsize);
    printf("Grid = %d\n", nside);
    printf("Maximum Radius = %6.1g\n", rmax);
    Float gridsize = rmax/(boxsize/nside);
    printf("Radius in Grid Units = %6.3g\n", gridsize);
    if (gridsize<1) printf("#\n# WARNING: grid appears inefficiently coarse\n#\n");
    printf("Bins = %d\n", NBIN);
    printf("Order = %d\n", ORDER);
    assert(ORDER<=MAXORDER);   // Actually, this will run, but it would give silent zeros.

    Particle *orig_p;
    if (make_random) {
	// If you want to just make random particles instead:
	assert(np>0);
	assert(boxsize==rescale);    // Nonsense if not!
	orig_p = make_particles(boxsize, np);
    } else {
	orig_p = read_particles(rescale, &np, fname);
	assert(np>0);
    }
    if (qinvert) invert_weights(orig_p, np);
    if (qbalance) balance_weights(orig_p, np);

    Float grid_density = (double)np/nside/nside/nside;
    printf("Average number of particles per grid cell = %6.2g\n", grid_density);
    printf("Average number of particles per max_radius ball = %6.2g\n",
	np*4.0*M_PI/3.0*pow(rmax/boxsize,3.0));
    if (grid_density<1) printf("#\n# WARNING: grid appears inefficiently fine.\n#\n");

    check_bounding_box(orig_p, np, boxsize, rmax);


    // Now ready to compute!
    // Sort the particles into the grid.
    Grid grid(orig_p, np, boxsize, nside);
    printf("# Done gridding the particles\n");
    printf("# %d particles in use, %d with positive weight\n", grid.np, grid.np_pos);
    printf("# Weights: Positive particles sum to %f\n", grid.sumw_pos);
    printf("#          Negative particles sum to %f\n", grid.sumw_neg);
    free(orig_p);

    smsave = smload = NULL;
    if (loadname!=NULL) smload = new StoreMultipoles(grid.np_pos);
    if (savename!=NULL) smsave = new StoreMultipoles(grid.np_pos);

    IOTime.Start();
    if (smload!=NULL) {
        smload->load(loadname);
	pairs[0].load(smload->xi0, smload->xi2);
	    // Put all of the previous work in thread 0
    }
    IOTime.Stop();

    zero_power();
    fflush(NULL);
    Prologue.Stop();

    // Everything above here takes negligible time.  This line is nearly all of the work.
    MultipoleTime.Start();
    compute_multipoles(&grid, rmax);
    printf("# Done counting the pairs\n");
    MultipoleTime.Stop();

    // Output the results
    Epilogue.Start();
    sum_power();
    printf("\n# Binned weighted pair counts, monopole and quadrupole\n");
    pairs[0].report_pairs();

    printf("\n# Multipole power\n");
    cpower[0].report_power();

    IOTime.Start();
    if (smsave!=NULL) {
	pairs[0].save(smsave->xi0, smsave->xi2);
        smsave->save(savename);
    }
    IOTime.Stop();

    if (smload!=NULL) delete smload;
    if (smsave!=NULL) delete smsave;
    Epilogue.Stop();
    TotalTime.Stop();
    printf("\n# Total Time: %4.1f s\n", TotalTime.Elapsed());
    printf("# Prologue: %6.3f s (%4.1f%%)\n", Prologue.Elapsed(), Prologue.Elapsed()/TotalTime.Elapsed()*100.0);
    printf("# Epilogue: %6.3f s (%4.1f%%)\n", Epilogue.Elapsed(), Epilogue.Elapsed()/TotalTime.Elapsed()*100.0);
    printf("# IO Time:  %6.3f s (%4.1f%%)\n", IOTime.Elapsed(), IOTime.Elapsed()/TotalTime.Elapsed()*100.0);
    printf("# Pairs:    %6.3f s (%4.1f%%)\n", MultipoleTime.Elapsed(), MultipoleTime.Elapsed()/TotalTime.Elapsed()*100.0);
    return 0;
}
