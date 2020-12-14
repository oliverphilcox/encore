# 3PCF-Estimator

C++ code for estimating the isotropic 3PCF multipoles for an arbitrary survey geometry. This is based on code by Daniel Eisenstein, implementing the algorithm of Slepian & Eisenstein (2015). 

The ./grid_multipoles code runs the analysis and the run_qpm.csh script is a SLURM run script. This requires a set of 32 random particle files with 1.5x more randoms than galaxies in each and negative weights. The periodic behavior is restored by the -DPERIODIC compiler option.
