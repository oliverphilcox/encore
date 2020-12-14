#!/bin/csh
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -t 0-03:00:01
#SBATCH --mem=10000
#SBATCH -o /home/ophilcox/out/3pcf_run.%a.out         # File to which STDOUT will be written
#SBATCH -e /home/ophilcox/out/3pcf_run.%a.err         # File to which STDERR will be written
#SBATCH --mail-type=END,FAIL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ophilcox@princeton.edu # Email to which notifications will be sent
#SBATCH --array=0-100

# We could access a task ID by SLURM_ARRAY_TASK_ID
set command = "./grid_multipoles -box 1000 -scale 1 -rmax 100 -ngrid 40"
echo $command

set inn=/tigress/ophilcox/quijote_3pcf
set root=quijote
set out=/tigress/ophilcox/quijote_3pcf/slepian_output_100_rsd
mkdir $out

set mockno=$SLURM_ARRAY_TASK_ID

echo Using mock $mockno

set outroot = $root.$mockno
set multfile = $out/$root.$mockno.mult

echo Running DDD counts
$command -in $inn/mock_$mockno -save $multfile > $out/$outroot.ddd.out
echo Done with DDD

echo Starting RRR
$command -in $inn/ran_00 -invert > $out/$outroot.rrr.out
echo Done with RRR

echo Starting NNN counts
foreach n ( 00 01 02 03 04 05 06 07 08 09 \
	    10 11 12 13 14 15 16 17 18 19 \
	    20 21 22 23 24 25 26 27 28 29 \
	    30 31 )
    # Make new file of Data+Randoms
    cp -f $inn/mock_$mockno $inn/tmp_ran_$n.$mockno
    cat $inn/ran_$n >> $inn/tmp_ran_$n.$mockno

    echo Starting NNN_$n
    $command -in $inn/tmp_ran_$n.$mockno -load $multfile -balance > $out/$outroot.r$n.out #nnn_$n.out

    # Remove trash
    rm $inn/tmp_ran_$n.$mockno
end

rm $multfile

echo Computations Complete
