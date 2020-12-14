#!/bin/csh
#SBATCH -n 36
#SBATCH -N 1
#SBATCH -t 0-04:00
#SBATCH -p itc_cluster
#SBATCH --mem=70000
#SBATCH -o Logs/dje_run.%A.%a.out         # File to which STDOUT will be written
#SBATCH -e Logs/dje_run.%A.%a.err         # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=deisenstein@cfa.harvard.edu # Email to which notifications will be sent

# We could access a task ID by SLURM_ARRAY_TASK_ID

setenv OMP_NUM_THREADS $SLURM_NPROCS

#set command = "time ./grid_multipolesAVX -box 4000 -scale 1 -rmax 180 -ngrid 100"
echo "Did you re-compile grid_multipoles for the correct NBIN?"
echo "Seeing $OMP_NUM_THREADS OMP threads and $SLURM_NPROCS slurm threads."

# Try to create a ramdisk, use it as $tmp

set tmp = /dev/shm/deisenstein
/bin/rm -rf $tmp       # Delete, just in case we have crud from a previous run.
mkdir $tmp


set script = script.patchy.csh
cp $script $tmp

mkdir $tmp/compile
cd /n/regal/eisenstein_lab/deisenstein/Grid
#tar xf /n/regal/eisenstein_lab/deisenstein/Grid/grid_multipoles.tar
    /bin/cp grid_multipoles.tar grid_multipoles.cpp spherical_harmonics.cpp \
    generateCartesianMultipolesASM.c \
    avxsseabrev.h externalmultipoles.h promote_numeric.h threevector.hh makefile \
    $tmp/compile
cd $tmp/compile
make
set command = "time $tmp/compile/grid_multipolesAVX -box 4000 -scale 1 -rmax 180 -ngrid 100"
echo command

cd /n/regal/eisenstein_lab/deisenstein/Grid
#make clean
#make

# We need to get a 4-digit number
set num = `printf %04d $SLURM_ARRAY_TASK_ID`

set in = Patchy6
set root = patchy6.dr12.mock.$num
set ranroot = patchy6.dr12
set outroot = $root
set out = /n/regal/eisenstein_lab/deisenstein/Grid/DR12.patchy.v02.big10
set tmpout = $tmp
set errfile = errlog.$num
set errlog = $out/$errfile

# If we want to put multiple outputs into the same directory, then 
# $outroot and $errfile need to be distinct.

#if (-e $out) 
#    echo "Directory $out exists"
#    exit
#endif

mkdir $out
rm -i $errlog
date > $errlog
echo Executing $0 >> $errlog
echo Executing job $num >> $errlog
echo $OMP_NUM_THREADS >> $errlog

set multfile = $tmp/$root.mult

# Get the data and run it, with storage
gunzip -c $in/$root.gz > $tmp/$root.data

echo Starting Computation
echo "Starting DDD" >> $errlog
date >> $errlog
($command -in $tmp/$root.data -save $multfile \
	> $tmpout/$outroot.ddd.out) >>& $errlog

echo Done with DDD


# Get a base random and run it, with inversion
gunzip -c $in/$ranroot.ran.00.gz > $tmp/$root.ran.00

echo "Starting RRR" >> $errlog
date >> $errlog
($command -in $tmp/$root.ran.00 -invert \
	> $tmpout/$outroot.rrr.out) >>& $errlog

echo Done with RRR

cat $tmp/$root.data >> $tmp/$root.ran.00

echo "Starting D-R check" >> $errlog
date >> $errlog
($command -in $tmp/$root.ran.00 \
	> $tmpout/$outroot.ch0.out) >>& $errlog

/bin/rm -f $tmp/$root.ran.0
echo Done with D-R 0 check

# Now make D-R and run that, with loading
#foreach n ( 00 01 02 03 04 05 06 07 08 09 \
#	    10 11 12 13 14 15 16 17 18 19 \
#	    20 21 22 23 24 25 26 27 28 29 \
#	    30 31 )
foreach n ( 00 01 02 03 04 05 06 07 08 09 )
    /bin/cp -f $tmp/$root.data $tmp/$root.ran.$n
    gunzip -c $in/$ranroot.ran.$n.gz >> $tmp/$root.ran.$n

    echo "Starting D-R $n" >> $errlog
    date >> $errlog
    ($command -in $tmp/$root.ran.$n -load $multfile \
	    > $tmpout/$outroot.r$n.out) >>& $errlog

    /bin/rm -f $tmp/$root.ran.$n
    echo Done with D-R $n

end    # foreach D-R loop

# Combine the files
module load hpc/python-2.7.3
python ./sumfiles.py $tmpout/$outroot.r??.out > $tmpout/$outroot.sum

echo Finished with computation.  Placing results into $out/$outroot.tgz
echo Finished with computation.  Placing results into $out/$outroot.tgz >> $errlog
date >> $errlog
pushd $tmpout
echo >> $errlog
/bin/ls -l >> $errlog
echo >> $errlog
tail $outroot.sum >> $errlog
/bin/cp $errlog .
tar cfz $outroot.tgz $outroot.*.out $outroot.sum $errfile $script
popd
/bin/mv $tmpout/$outroot.tgz $tmpout/$outroot.sum $out

# Destroy ramdisk
/bin/rm -rf $tmp
