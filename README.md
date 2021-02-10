# encore: Efficient N-point Correlator Estimation

C++ code for estimating the isotropic NPCF multipoles for an arbitrary survey geometry in O(N^2) time. This is based on code by Daniel Eisenstein, implementing the algorithm of Slepian et al. (in prep.), and uses their conventions. This currently features support for the isotropic 2PCF, 3PCF, 4PCF, 5PCF and 6PCF, and is a sister code to the [python implementation](https://github.com/oliverphilcox/pynpcf), which contains only 3PCF and 4PCF algorithms. For the 4PCF, 5PCF and 6PCF algorithms, the runtime is dominated by sorting the spherical harmonics into bins, which has complexity O(N_galaxy x N_bins^3 x N_ell^5) [4PCF], O(N_galaxy x N_bins^4 x N_ell^8) [5PCF] or O(N_galaxy x N_bins^5 x N_ell^11) [6PCF]. We caution that the higher-point functions will be necessarily slow to compute unless N_bins and N_ell are small.

#### Requirements:
- C++ compiler (tested with g++ 4.8.5)
- *(Optional)*: AVX compatibility.
- *(Optional)*: OpenMP for multiprocessing.
- *(Optional*): Python (tested with 2.7 & 3.6) with numpy and sympy installed for file summation and edge correction.

#### Authors:
- Oliver Philcox (Princeton / IAS, [ohep2@cantab.ac.uk](mailto:ohep2@cantab.ac.uk))
- Zachary Slepian (Florida)
- Daniel Eisenstein (Harvard)

## Usage
- To run the code, first compile it using ```make clean; make```. You will need to edit the Makefile depending on your particular configurations. In particular, the Makefile has the options ```-DOPENMP``` to run with OpenMP support for parallization, ```-DFOURPCF```, ```-DFIVEPCF``` and ```-DSIXPCF``` to enable the 4PCF/5PCF/6PCF computation, ```-DPERIODIC``` to assume a periodic box geometry, and ```-DAVX``` to additionally compile the code using AVX instructions. The 2PCF and 3PCF are always computed.
- The main pair-counting code can then be run using ```./encore``` or ```./encoreAVX```, if your machine has support for AVX instruction sets. We recommend the latter option when running on clusters, as it gives significant speed-boosts for the 3PCF algorithm, for ell-max > 4. The code takes a number of input command-line options, described below.
- For power-users, we also provide the ```run_npcf.csh``` script, which automates computation of the full NPCF (including edge-corrections), given data and a set of random catalogs. This can be run either from the terminal or with SLURM, and further documentation is provided at the top of the module. Assuming ell-max = 5, this is sufficient to compute the full 2PCF, 3PCF and 4PCF of BOSS CMASS-North in around 2 hours on a 16-core machine.
- Two key parameters are hard-coded in the ```encore.cpp``` file for memory allocation reasons. These are ```ORDER``` (maximum ell used for the spherical harmonics) and ```NBIN``` (total number of linearly-spaced radial bins, at least ```N-1``` for the ```N``` point function). There's also ```MAXTHREAD```, but that probably doesn't need to be changed. The code must be recompiled after changing any of these options.
- The code currently has support for all ell up to ```ell_max=10``` for the 3PCF and 4PCF, ```ell_max=5``` for the 5PCF, and ```ell_max=3``` for the 6PCF. If necessary, higher multipoles can be added by running the ```coupling_weights.py``` script and changing the ```MAXORDER``` parameters in the C++ code.
- The output products are stored in the ```output/``` directory as ```.txt``` files. The format is described in the individual files: in general, they are arrays with the column and row specifying the radial and angular bin respectively. Note that we only store parity even correlators with multipoles satisfying the triangle conditions and with ```BIN1 < BIN2 < BIN3``` etc.
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
