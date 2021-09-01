#!/bin/csh

##################### DOCUMENTATION #####################
### Shell script for running the encore NPCF-estimator function on a GRF sim.
# The sims are computed using the generate_grfs.py python script.
# This gives an unbiased NPCF estimate, but the covariance will have enhanced shot-noise (due to the way the GRFs are created).
# We only need to compute D counts here and do not perform edge corrections.
# The code should be compiled in *PERIODIC* mode.
#
# This can be run either from the terminal or as a SLURM script (using the below parameters).
# The code should be compiled (with the relevant options, i.e. N-bins, ell-max and 3PCF/4PCF/5PCF/6PCF) before this script is run.
# The script should be run from the code directory
# The output will be a set of .zeta_{N}pcf.txt files in the specified directory as well as a .tgz compressed directory of other intermediary outputs
#
# NB: If needed, we could access a task ID by SLURM_ARRAY_TASK_ID, if we're running with SLURM
##########################################################

#SBATCH -n 16 # cpus
#SBATCH -N 1 # tasks
#SBATCH -t 0-02:30:59 # time
#SBATCH --mem-per-cpu=2GB
#SBATCH -o /home/ophilcox/out/grf_npcf_run.%a.out         # File to which STDOUT will be written (make sure the directory exists!)
#SBATCH -e /home/ophilcox/out/grf_npcf_run.%a.err         # File to which STDERR will be written
#SBATCH --mail-type=END,FAIL         # Type of email notification
#SBATCH --mail-user=ophilcox@princeton.edu # Email to which notifications will be sent
#SBATCH --array=0-99

set rmin = 0 # minimum radius in Mpc/h
set rmax = 170 # maximum radius in Mpc/h

# Other inputs
set scale = 1 # rescaling for co-ordinates
set boxsize = 1574
set ngrid = 50 # grid-size for accelerating pair count
set useAVX = 1 # whether to use AVX

# Mock numbers to use
set batch_size = 50
@ mock_min = ( $SLURM_ARRAY_TASK_ID * $batch_size )
@ mock_max = ( $mock_min + $batch_size )

# File directories
set in = /projects/QUIJOTE/Oliver/npcf/rsd_grf/ # input directory (see above for required contents)

##########################################################

# Set number of threads (no SLURM)
# set OMP_NUM_THREADS = 16

# Set number of threads (with SLURM)
setenv OMP_NUM_THREADS $SLURM_NPROCS

# Define the simulations
module load anaconda3
conda activate ptenv

date
python generate_grfs.py $mock_min $mock_max

# Define command to run the C++ code
if ($useAVX) then
  set code = ./encoreAVX
else
  set code = ./encore
endif

set command = "$code -rmax $rmax -rmin $rmin -ngrid $ngrid  -scale $scale -boxsize $boxsize"

date

# Iterate over the GRF fields
@ n = $mock_min
while ($n <= $mock_max)

  echo "Starting grf $n"

  $command -in $in/mocks/rsd_grf$n.txt -outstr grf_$n

  # Copy the output into the temporary directory
  mv output/grf_${n}_?pcf.txt $in/

  # Remove the mock
  rm $in/mocks/rsd_grf$n.txt

  @ n += 1
end

date
