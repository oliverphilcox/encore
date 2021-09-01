#!/bin/csh

##################### DOCUMENTATION #####################
### Shell script for running the encore NPCF-estimator function on a lognormal sim. (Oliver Philcox, 2021)
# The sims are computed using the python/generate_lognormal.py python script.
# This can be run either from the terminal or as a SLURM script (using the below parameters).
# The code should be compiled (with the relevant options, i.e. N-bins, ell-max and 4PCF/5PCF/6PCF) before this script is run. The isotropic 2PCF and 3PCF will always be computed.
# The script should be run from the code directory
# The code should be compiled in *PERIODIC* mode, and does not perform any edge corrections (since these are trivial).
#
# This script will compute the D^N counts, the (D-R)^N counts for 32 random subsets, and the R^N counts for one subset (should be sufficient).
# The output will be a set of .zeta_{N}pcf.txt files in the specified directory as well as a .tgz compressed directory of other intermediary outputs
# We also compute the disconnected 4PCF piece, saved as zeta_discon_4pcf.txt in the same directory.
#
# NB: If needed, we could access a task ID by SLURM_ARRAY_TASK_ID, if we're running with SLURM
##########################################################

#SBATCH -n 16 # cpus
#SBATCH -N 1 # tasks
#SBATCH -t 0-02:58:59 # time
#SBATCH --mem-per-cpu=2GB
#SBATCH -o /home/ophilcox/out/lognormal_npcf_run.%a.out         # File to which STDOUT will be written (make sure the directory exists!)
#SBATCH -e /home/ophilcox/out/lognormal_npcf_run.%a.err         # File to which STDERR will be written
#SBATCH --mail-type=END,FAIL         # Type of email notification
#SBATCH --mail-user=ophilcox@princeton.edu # Email to which notifications will be sent
#SBATCH --array=0-24

set rmin = 20 # minimum radius in Mpc/h
set rmax = 160 # maximum radius in Mpc/h

# Other inputs
set scale = 1 # rescaling for co-ordinates
set boxsize = 1574
set ngrid = 50 # grid-size for accelerating pair count
set useAVX = 1 # whether to use AVX

# Mock numbers to use per batch
set batch_size = 4
@ mock_min = ( $SLURM_ARRAY_TASK_ID * $batch_size )
@ mock_max = ( $mock_min + $batch_size - 1)

# File directories
set in = /projects/QUIJOTE/Oliver/npcf/rsd_lognormal/mocks # input directory (see above for required contents)
set tmp = /scratch/gpfs/ophilcox/lognormal_$SLURM_ARRAY_TASK_ID
set ranroot = lognormal
set out = /projects/QUIJOTE/Oliver/npcf/rsd_lognormal

# Load some python environment with numpy and sympy installed
module load anaconda3
conda activate ptenv

##########################################################

# Set number of threads (no SLURM)
#set OMP_NUM_THREADS = 16

# Set number of threads (with SLURM)
setenv OMP_NUM_THREADS $SLURM_NPROCS

# Define command to run the C++ code
if ($useAVX) then
  set code = ./encoreAVX
else
  set code = ./encore
endif

set command = "$code -rmax $rmax -rmin $rmin -ngrid $ngrid  -scale $scale -boxsize $boxsize"

# Create a temporary directory for saving
/bin/rm -rf $tmp       # Delete, just in case we have crud from a previous run.
mkdir $tmp

# Copy this script in for posterity
cp run_npcf_lognormal.csh $tmp

# Create output directory
if (!(-e $out)) then
    mkdir $out
endif

# Create an output file for errors
set errfile = errlog_$SLURM_ARRAY_TASK_ID
set errlog = $out/$errfile
set tmpout = $tmp
rm -f $errlog
date > $errlog
echo Executing $0 >> $errlog
echo $command >> $errlog
echo $OMP_NUM_THREADS >> $errlog

date

# Define the simulations
date
python python/generate_lognormal.py $mock_min $mock_max

# Iterate over the GRF fields
@ nn = $mock_min
while ($nn <= $mock_max)

  echo
  echo "Starting lognormal simulation $nn"
  echo

  set sim_no = $nn
  set root = lognormal$nn

  # Filename for saved multipoles (a big file)
  set multfile = $tmp/$root.mult

  # Extract the data into our temporary ramdisk
  gunzip -c $in/$root.data.gz > $tmp/$root.data

  # Find number of galaxies (needed later for R^N periodic counts)
  set Ngal = `cat $tmp/$root.data | wc -l`
  set Ngal = `expr $Ngal + 1`

  #### Compute D^N NPCF counts
  # Note that we save the a_lm multipoles from the data here
  echo Starting Computation
  echo "Starting D^N" >> $errlog
  date >> $errlog
  ($command -in $tmp/$root.data -save $multfile -outstr $root.data > $tmpout/$root.d.out) >>& $errlog
  # Remove the output - we don't use it
  rm output/$root.data_?pcf.txt
  rm output/$root.data_2pcf_mult?.txt

  echo "Done with D^N"

  ### Compute R^N NPCF counts
  # We don't really need to do this each iteration, but it's not expensive
  # We just use one R catalog for this and invert it such that the galaxies are positively weighted
  gunzip -c $in/$ranroot.ran.00.gz > $tmp/$root.ran.00

  echo "Starting R^N" >> $errlog
  date >> $errlog
  ($command -in $tmp/$root.ran.00 -outstr $root.r -invert > $tmpout/$root.r.out) >>& $errlog
  # Copy the output into the temporary directory, includind disconnected pieces
  mv output/$root.r_?pcf.txt $tmpout/
  mv output/$root.r_2pcf_mult?.txt $tmpout/

  echo "Done with R^N"

  Now make D-R for each of 32 random catalogs, with loading
  foreach n ( 00 01 02 03 04 05 06 07 08 09 \
  	    10 11 12 13 14 15 16 17 18 19 \
  	    20 21 22 23 24 25 26 27 28 29 \
  	    30 31 )

      # First copy the randoms and add the data
      /bin/cp -f $tmp/$root.data $tmp/$root.ran.$n
      gunzip -c $in/$ranroot.ran.$n.gz >> $tmp/$root.ran.$n

      ### Compute the (D-R)^N counts
      # This uses the loaded data multipoles from the D^N step
      # Note that we balance the weights here to ensure that Sum(D-R) = 0 exactly
      echo "Starting D-R $n" >> $errlog
      date >> $errlog
      ($command -in $tmp/$root.ran.$n -load $multfile -outstr $root.n$n -balance > $tmpout/$root.n$n.out) >>& $errlog
      # Copy the output into the temporary directory, including disconnected pieces
      mv output/$root.n${n}_?pcf.txt $tmpout/
      mv output/$root.n${n}_2pcf_mult?.txt $tmpout/

      # Remove the random catalog
      /bin/rm -f $tmp/$root.ran.$n
      echo Done with D-R $n

  end
  # foreach D-R loop

  echo "Combining files together without performing edge-corrections (using analytic R^N counts)"
  python python/combine_files_periodic.py $tmpout/$root $Ngal $boxsize $rmin $rmax >>& $errlog

  echo "Combining disconnected files together without performing edge-corrections (using analytic R^N counts)"
  python python/combine_disconnected_files_periodic.py $tmpout/$root 4 $Ngal $boxsize $rmin $rmax >>& $errlog

  # Do some cleanup
  rm $tmp/$root.data $multfile

  # Now move the output files into the output directory.
  # Compress all the auxilliary files and copy
  echo Finished with computation.  Placing results into $out/
  echo Finished with computation.  Placing results into $out/ >> $errlog
  date >> $errlog
  pushd $tmpout > /dev/null
  echo >> $errlog
  /bin/ls -l >> $errlog
  /bin/cp $errlog .
  tar cfz $root.tgz $root.*.out $root.*pcf.txt $root.*2pcf_mult?.txt $errfile run_npcf_lognormal.csh
  popd > /dev/null
  /bin/mv $tmpout/$root.tgz $tmpout/$root.zeta_*pcf.txt $out/

  # Remove the mock
  rm $in/lognormal$nn.data.gz

  @ nn += 1
end

# Destroy ramdisk
/bin/rm -rf $tmp

date
