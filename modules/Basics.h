#ifndef BASICS_H
#define BASICS_H

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
    Float3 rect_boxsize;   // 3D dimensions of the periodic volume
    int ncells;       // Grid size (per volume)
    Cell *c;		// The list of cells
    Float cellsize;   // Size of one cell
    Float max_boxsize; // largest dimension of cuboid box
    Float3 shift; // vector to shift particles by such that they are in the primary box
    Particle *p;	// Pointer to the list of particles
    int np;		// Number of particles
    integer3 nside_cuboid; // number of cells along each dimension of cuboidal box
    int nf;     // number of filled cells
    int np_pos;		// Number of particles
    int *pid;		// The original ordering
    int *filled;    // List of filled cells
    Float sumw_pos, sumw_neg; // Summing the weights

    int test_cell(integer3 cell){
    // returns -1 if cell is outside the grid or wraps around for the periodic grid
#ifndef PERIODIC
    	if(nside_cuboid.x<=cell.x||cell.x<0||nside_cuboid.y<=cell.y||cell.y<0||nside_cuboid.z<=cell.z||cell.z<0){
            return -1;
        }else
#endif
    		return wrap_cell(cell);
    }

    int wrap_cell(integer3 cell) {
        // Return the 1-d cell number, after wrapping
	// We apply a very large bias, so that we're
	// guaranteed to wrap any reasonable input.
#ifdef PERIODIC
	int cx = (cell.x+ncells)%nside_cuboid.x;
	int cy = (cell.y+ncells)%nside_cuboid.y;
	int cz = (cell.z+ncells)%nside_cuboid.z;
	// return (cx*nside+cy)*nside+cz;
	int answer = (cx*nside_cuboid.y+cy)*nside_cuboid.z+cz;
#else
    int answer = (cell.x*nside_cuboid.y+cell.y)*nside_cuboid.z+cell.z;
#endif
    assert(answer<ncells&&answer>=0);

    return answer;
    }

    integer3 cell_id_from_1d(int n) {
	// Undo 1d back to 3-d indexing
        assert(n>=0&&n<ncells);
	integer3 cid;
	cid.z = n%nside_cuboid.z;
	n = n/nside_cuboid.z;
	cid.y = n%nside_cuboid.y;
	cid.x = n/nside_cuboid.y;
	return cid;
    }

    int pos_to_cell(Float3 pos) {
        // Return the 1-d cell number for this position, properly wrapped
	// We assume the first cell is centered at cellsize/2.0
	//return wrap_cell( floor3(pos/cellsize+Float3(0.5,0.5,0.5)));
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
        free(filled);
        free(c);
        return;
    }

    Grid(Particle *input, int _np, Float3 _rect_boxsize, Float _cellsize, Float3 _shift) {
	// The constructor: the input set of particles is copied into a
	// new list, which is ordered by cell.
	// After this, Grid is self-sufficient; one could discard *input
    rect_boxsize = _rect_boxsize;
    cellsize = _cellsize;
    shift = _shift;
	np = _np;
	np_pos = 0;
    max_boxsize=fmax(rect_boxsize.x,fmax(rect_boxsize.y,rect_boxsize.z));
	assert(max_boxsize>0&&np>=0);

    nside_cuboid = integer3(ceil3(rect_boxsize/cellsize));

	ncells = nside_cuboid.x*nside_cuboid.y*nside_cuboid.z;

    p = (Particle *)malloc(sizeof(Particle)*np);
	pid = (int *)malloc(sizeof(int)*np);
	printf("# Allocating %6.3f MB of particles\n", (sizeof(Particle)+sizeof(int))*np/1024.0/1024.0);
	c = (Cell *)malloc(sizeof(Cell)*ncells);
	printf("# Allocating %6.3f MB of cells\n", (sizeof(Cell))*ncells/1024.0/1024.0);

	// Now we want to copy the particles, but do so into grid order.
	// First, figure out the cell for each particle
    // Shift them to the primary volume first
	int *cell = (int *)malloc(sizeof(int)*np);

#ifdef PERIODIC
    for (int j=0; j<np; j++) cell[j] = pos_to_cell(input[j].pos);
#else
    for (int j=0; j<np; j++) cell[j] = pos_to_cell(input[j].pos - shift);
#endif
    // Histogram the number of particles in each cell
	int *incell = (int *)malloc(sizeof(int)*ncells);
	for (int j=0; j<ncells; j++) incell[j] = 0.0;
	for (int j=0; j<np; j++) incell[cell[j]]++;

    // Create list of filled cells
    nf=0;
    for (int j=0; j<ncells; j++) if(incell[j]>0) nf++;
    filled = (int *)malloc(sizeof(int)*nf);
    for (int j=0,k=0; j<ncells; j++) if(incell[j]>0) filled[k++]=j;

    printf("\nThere are %d filled cells compared with %d total cells.\n",nf,ncells);

	// Count the number of positively weighted particles
	sumw_pos = sumw_neg = 0.0;
	for (int j=0; j<np; j++){
	    if (input[j].w>=0) {
	    	np_pos++;
		sumw_pos += input[j].w;
	    } else {
		sumw_neg += input[j].w;
	    }
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
#ifdef PERIODIC
        // Switch to cell-centered positions
	    p[index].pos = cell_centered_pos(input[j].pos);
#endif
	    pid[index] = j;	 // Storing the original index

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
#endif
