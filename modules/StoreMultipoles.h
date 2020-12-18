#ifndef STORE_MULTIPOLES_H
#define STORE_MULTIPOLES_H

// STORAGE NOTES (D. J. Eisenstein)
//
// As an advanced option, we can store and reload the multipoles per
// particle and the binned pair counts.  The intention is to support
// use cases in which one has a particle list with data particles
// (weight >=0) and random particles (weight < 0).  One first runs the
// code with only the data particles, saving the output.  One then
// re-runs with a file that has some random particles postpended to
// the input file.  By loading the stored file, the code will skip the
// investigation of any pairs of positively weighted particles, thereby
// re-using all of the DD work (and hence the DDD three-point).
//
// The intention is that one might rerun the code many times with
// different sets of random points.  Doing this will build up statistics
// on the DR and RR counts.  By having each run use only n_R similar
// to n_D, we avoid doing far too much work on RR as opposed to DR.
// The optimum is to use 1.5--2 times more randoms than data in each run.
//
// In this mode, only non-negative weighted primary particles have any
// information stored, although they will search all secondary particles
// when the stored file is created.  When re-loaded, only pairs of
// non-negative primaries *and* secondaries are skipped.  So the initial
// run should only have non-negative particles, or the book-keeping
// will go askew.
//
// The stored files are in a pure binary format; read the source code if you need
// to parse it.  Note that this file isn't small: NBIN*NMULT*nparticle doubles!
// For ORDER=4 and NBIN=10, that's 2800 bytes per particle.
// For ORDER=10 and NBIN=10, 17600 bytes per particle.
// Hopefully we can read it faster than we can recompute the DD!
// BOSS-scaled DD appears to be about 5000 primaries/second on a 6-core machine,
// so we need >1e8 Mbyte/sec to make this worthwhile.

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

#endif
