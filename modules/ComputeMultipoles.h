#ifndef COMPUTE_MULTIPOLES_H
#define COMPUTE_MULTIPOLES_H

// ====================  Computing the pairs ==================

void compute_multipoles(Grid *grid, Float rmin, Float rmax) {
    int maxsep = ceil(rmax/grid->cellsize);   // Maximum distance we must search
    int ne;
    Float rmax2 = rmax*rmax;
    Float rmin2 = rmin*rmin; //rmax2*1e-12;    // Just an underflow guard
    uint64 cnt = 0;

    Multipoles *mlist = new Multipoles[MAXTHREAD*NBIN];  // Set up all of this space

    // Easy to multi-thread this top loop!
    // But some cells have trivial amounts of work, so we will first make a list of the work.
    // Including the empty cells appears to fool the dynamic thread allocation sometimes.

    STimer accmult, powertime; // measure the time spent accumulating powers for multipoles
    // We're going to loop only over the non-empty cells.
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic,8) reduction(+:cnt)
#endif
    for (ne=0; ne<grid->nf; ne++) {
    int n = grid->filled[ne];  // Fetch the cell number

    // Decide which thread we are in
#ifdef OPENMP
	int thread = omp_get_thread_num();
        assert(omp_get_num_threads()<=MAXTHREAD);
        if (ne==0) printf("# Running on %d threads.\n", omp_get_num_threads());
#else
	int thread = 0;
        if (ne==0) printf("# Running single threaded.\n");
#endif
    if(int(ne%1000)==0) printf("Computing cell %d of %d on thread %d\n",ne,grid->nf,thread);
#ifdef FIVEPCF
    else if (int(ne%100)==0) printf("Computing cell %d of %d on thread %d\n",ne,grid->nf,thread);
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
      if(thread==0) accmult.Start();
    	for (delta.x = -maxsep; delta.x <= maxsep; delta.x++)
	    for (delta.y = -maxsep; delta.y <= maxsep; delta.y++)
	    for (delta.z = -maxsep; delta.z <= maxsep; delta.z++) {
		const int samecell = (delta.x==0&&delta.y==0&&delta.z==0)?1:0;

        // Check that the cell is in the grid!
        int tmp_test = grid->test_cell(prim_id+delta);
        if(tmp_test<0) continue;
        Cell sec = grid->c[tmp_test];

        // Define primary position
        Float3 ppos = grid->p[j].pos;
#ifdef PERIODIC
        ppos-=grid->cell_sep(delta);
#endif
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
		    // Check if this is in the correct binning ranges
            if (norm2<rmax2 && norm2>rmin2) cnt++; else continue;

		    // Now what do we want to do with the pair?
		    norm2 = sqrt(norm2);  // Now just radius
		    // Find the radial bin
		    int bin = floor((norm2-rmin)/(rmax-rmin)*NBIN);

        // Define x/r,y/r,z/r
		    dx = dx/norm2;

        //continue;   // Skip pairs and multipoles

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
      if(thread==0) accmult.Stop();

	    if (smsave && grid->p[j].w>=0) {
	        // We're saving multipoles, and this particle has positive weight.
		int pid = grid->pid[j];
		for (int b=0; b<NBIN; b++) mult[b].save(smsave->fetchM(pid,b), smsave->fetchC(pid,b));
	    }

	    // Now add these multipoles into the cross-powers
	    // This step takes very little time for the 3PCF, but is time-limiting for higher-point functions.
      if(thread==0) powertime.Start();
	    npcf[thread].add_to_power(mult, primary_w);
      if(thread==0) powertime.Stop();
// #ifdef FIVEPCF
// if (int(ne%100)==0) printf("Powertime: %6.3f\n", powertime.Elapsed());
// #endif

	} // Done with this primary particle
    } // Done with this primary cell, end of omp pragma
    if (_gpumode > 0) {
      if (!_gpumemcpy) npcf[0].do_copy_memory();  //if not memcpy, must now copy back to host
      npcf[0].free_gpu_memory(); //free all GPU memory
    }

#ifndef OPENMP
#ifdef AVX
    printf("\n# Time to compute required powers of x_hat, y_hat, z_hat (with AVX): %.2f\n\n",accmult.Elapsed());
#else
  printf("\n# Time to compute required powers of x_hat, y_hat, z_hat (no AVX): %.2f\n\n",accmult.Elapsed());
#endif
#endif

    printf("# We counted  %lld pairs within [%f %f].\n", cnt, rmin, rmax);
    printf("# Average of %f pairs per primary particle.\n",
    		(Float)cnt/grid->np);
    Float3 boxsize = grid->rect_boxsize;
    float expected = grid->np * (4*M_PI/3.0)*(pow(rmax,3.0)-pow(rmin,3.0))/(boxsize.x*boxsize.y*boxsize.z);
    printf("# We expected %1.0f pairs per primary particle, off by a factor of %f.\n", expected, cnt/(expected*grid->np));

    delete[] mlist;

    // Detailed timing breakdown
    printf("\n# Accumulate Powers: %6.3f s\n", accmult.Elapsed());
    printf("# Compute Power: %6.3f s\n\n", powertime.Elapsed());

    return;
}

#endif
