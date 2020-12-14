#ifndef STORE_MULTIPOLES_H
#define STORE_MULTIPOLES_H

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
