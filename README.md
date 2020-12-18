# NPCF-Estimator

C++ code for estimating the isotropic NPCF multipoles for an arbitrary survey geometry in O(N^2) time. This is based on code by Daniel Eisenstein, implementing the algorithm of Slepian et al. (in prep.), and uses their conventions. This is a sister code to the [python implementation](https://github.com/oliverphilcox/pynpcf), and currently features support for the 3PCF and 4PCF.

After compiling the code, via ```make clean; make```, the code is run using ```./npcf_estimator``` or ```./npcf_estimatorAVX``` if your machine has support for AVX instruction sets. This is usually run on both sets of (data - randoms) [usually with 1.5x more randoms than galaxies in each and negative weights for the randoms such that the total weight is zero] and the randoms themselves. The periodic behavior is restored by the  ```-DPERIODIC``` compiler option. If ```-DFOURPCF``` when the code is compiled, the 4PCF will be computed alongside the 3PCF.

Requirements:
- C++ compiler (tested with g++4.8.5)
- *(Optional)*: AVX compatibility.

Authors:
- Oliver Philcox (Princeton)
- Zachary Slepian (Florida)
- Daniel Eisenstein (Harvard)

## Taken from the header:


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
