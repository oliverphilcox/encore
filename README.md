# NPCF-Estimator

C++ code for estimating the isotropic NPCF multipoles for an arbitrary survey geometry in O(N^2) time. This is based on code by Daniel Eisenstein, implementing the algorithm of Slepian et al. (in prep.), and uses their conventions. This currently features support for the 3PCF, 4PCF and 5PCF, and is a sister code to the [python implementation](https://github.com/oliverphilcox/pynpcf), which contains only 3PCF and 4PCF algorithms. For the 4PCF and 5PCF algorithms, the runtime is dominated by sorting the spherical harmonics into bins, which has complexity O(N_galaxy x N_bins^3 x N_ell^5) [4PCF] or O(N_galaxy x N_bins^4 x N_ell^8) [5PCF].

#### Requirements:
- C++ compiler (tested with g++ 4.8.5)
- *(Optional)*: AVX compatibility.
- *(Optional)*: OpenMP for multiprocessing.
- *(Optional*): Python (tested with 2.7 & 3.6) with numpy and sympy installed for file summation and edge correction.

#### Authors:
- Oliver Philcox (Princeton / IAS)
- Zachary Slepian (Florida)
- Daniel Eisenstein (Harvard)

## Usage
- To run the code, first compile it using ```make clean; make```. You will need to edit the Makefile depending on your particular configurations. In particular, the Makefile has the options ```-DOPENMP``` to run with OpenMP support for parallization, ```-DFOURPCF``` and ```-DFIVEPCF``` to enable the 4PCF/5PCF computation, ```-DPERIODIC``` to assume a periodic box geometry, and ```-DAVX``` to additionally compile the code using AVX instructions.
- The main pair-counting code can then be run using ```./npcf_estimator``` or ```./npcf_estimatorAVX```, if your machine has support for AVX instruction sets. We recommend the latter option when running on clusters, as it gives significant speed-boosts for the 3PCF algorithm, for ell-max > 4. The code takes a number of input command-line options, described below.
- For power-users, we also provide the ```run_npcf.csh``` script, which automates computation of the full NPCF (including edge-corrections), given data and a set of random catalogs. This can be run either from the terminal or with SLURM, and further documentation is provided at the top of the module. Assuming ell-max = 5, this is sufficient to compute the full 3PCF and 4PCF of BOSS CMASS-North in around 2 hours on a 16-core machine.
- Two key parameters are hard-coded in the ```npcf_estimator.cpp``` file for memory allocation reasons. These are ```ORDER``` (maximum ell used for the spherical harmonics) and ```NBIN``` (total number of linearly-spaced radial bins). There's also ```MAXTHREAD```, but that probably doesn't need to be changed. The code must be recompiled after changing any of these options.
- The code has support for all ell up to ```MAXORDER=10``` for the 3PCF and 4PCF, and ```MAXORDER=4``` for the 5PCF.
- The output products are stored in the ```output/``` directory as ```.txt``` files. The format is described in the individual files: in general, they are arrays with the column and row specifying the radial and angular bin respectively.
- Generally, one will run the code on ~ 30 (data-random) files, the corresponding (random-random) files, then combine to obtain the NPCF estimates. For (data-random) inputs, the randoms should have negative weights, such that the total summed weight is zero. The weights can be balanced and inverted using the optional parameters.
- For advanced usage, there is an option to store the multipoles of positively weighted primary particles. If multiple (D-R) sets are computed in series, this avoids the multipoles of the data being recomputed each time. More information regarding this is found in the comment in the ```modules/StoreMultipoles.h``` script.

#### Main Options:
- ```-rmax```: Maximum radius of the largest pairwise separation bin in Mpc/h (default: 200).
- ```-in```: Filename of the input file. This should be a space or tab-separated CSV file with columns [x,y,z,weight]. Lines starting with # will be skipped. (default: "sample.dat").
- ```-outstr```: String to prepend to the output files (default: "sample").
- ```-save```: Filename of the file in which to store the multipoles of positively weighted particles. If none is specified, these will not be stored. Note that this is a large binary file. (default: None).
- ```-load```: Filename of the file in which to load the multipoles of positively weighted particles from a previous run (default: None).

#### Other Options:
- ```-def```: This allows one to accept the default values for each parameter without giving other entries.
- ```-balance```: If set, rescale the negative weights such that the total weight is zero (useful for D-R pair counts).
- ```-invert```: If set, multiply all the weights by -1 (useful for R pair counts).
- ```-nside```: Gridsize used to accelerate the particle search . Overly large grid cells are inefficient at rejecting cells outside ```rmax``` and overly small cells incur more overhead. We recommend having several grid cells per rmax (default: 50).
- ```-ran```: Integer: if specified, ignore any input file and throw this many points in a cube at random.
- ```-box```: If drawing particles at random, this sets the periodic size of the computational domain (default: 400).
- ```-scale```: Dilate the input positions by this factor (default: 1).
