#ifndef DRIVER_H
#define DRIVER_H

// ====================  The Driver ===========================

Particle *make_particles(Float3 rect_boxsize, int np) {
    // Make np random particles
    srand48(1);      // For reproducibility
    Particle *p = (Particle *)malloc(sizeof(Particle)*np);
    for (int j=0; j<np; j++) {
        p[j].pos.x = drand48()*rect_boxsize.x;
        p[j].pos.y = drand48()*rect_boxsize.y;
        p[j].pos.z = drand48()*rect_boxsize.z;
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
        if (line[0]=='\n') continue;
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

void compute_bounding_box(Particle *p, int np, Float3 &rect_boxsize, Float &cellsize, Float rmax, Float3& pmin, int nside) {
    // Check that the bounding box is reasonable
    // This updates the boxsize and cellsize if we have a non-periodic box and the PERIODIC flag is not set
    Float3 pmax;
    pmin.x = pmin.y = pmin.z = 1e30;
    pmax.x = pmax.y = pmax.z = -1e30;
    for (int j=0; j<np; j++) {
        pmin.x = fmin(pmin.x, p[j].pos.x);
        pmin.y = fmin(pmin.y, p[j].pos.y);
        pmin.z = fmin(pmin.z, p[j].pos.z);
        pmax.x = fmax(pmax.x, p[j].pos.x);
        pmax.y = fmax(pmax.y, p[j].pos.y);
        pmax.z = fmax(pmax.z, p[j].pos.z);
    }
    printf("# Range of x positions are %6.2f to %6.2f\n", pmin.x, pmax.x);
    printf("# Range of y positions are %6.2f to %6.2f\n", pmin.y, pmax.y);
    printf("# Range of z positions are %6.2f to %6.2f\n", pmin.z, pmax.z);
    Float3 prange = pmax-pmin;
    Float biggest = prange.x;
    biggest = fmax(biggest, prange.y);
    biggest = fmax(biggest, prange.z);
    printf("# Biggest range is %6.2f\n", biggest);
    Float max_boxsize;

    if (prange.x>0.99*biggest && prange.y>0.99*biggest && prange.z>0.99*biggest) {
        // Probably using a cube of inputs, intended for a periodic box
#ifndef PERIODIC
    	fprintf(stderr,"#\n# WARNING: cubic input detected but you have not compiled with PERIODIC flag!\n#\n");
    	printf("#\n# WARNING: cubic input detected but you have not compiled with PERIODIC flag!\n#\n");
        // Do not alter the boxsize here
        printf("# Keeping periodic box-size at %6.2f\n", rect_boxsize.x);
        cellsize = rect_boxsize.x/nside; // compute cell width
#endif
    } else{
        // Probably a non-periodic input (e.g. a real dataset)
#ifdef PERIODIC
    	fprintf(stderr,"#\n# WARNING: non-cubic input detected but you have compiled with PERIODIC flag!\n#\n");
    	printf("#\n# WARNING: non-cubic input detected but you have compiled with PERIODIC flag!\n#\n");
#else
        // set max_boxsize to just enclose the biggest dimension plus r_max
        // NB: We natively wrap the grid (to allow for any position of the center of the grid)
        // Must add rmax to biggest to ensure there is no periodic overlap in this case.
        max_boxsize = 1.05*(biggest+rmax);
        cellsize = max_boxsize/nside; // compute the width of each cell
        // Now compute the size of the box in every dimension
        rect_boxsize = ceil3(prange/cellsize)*cellsize; // to ensure we fit an integer number of cells in each direction
        printf("# Setting non-periodic box-size to {%6.2f,%6.2f,%6.2f}\n", rect_boxsize.x,rect_boxsize.y,rect_boxsize.z);
#endif
    }

#ifdef PERIODIC
    max_boxsize = rect_boxsize.x;
    cellsize = max_boxsize/nside;
    if (biggest>max_boxsize*1.001)
	printf("#\n# WARNING: particles will overlap on period wrapping!\n#\n");
    if (biggest+rmax<max_boxsize*0.6)
	printf("#\n# WARNING: box periodicity seems too generous, will hurt grid efficiency!\n#\n");
#endif



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

#endif
