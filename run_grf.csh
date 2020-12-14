#!/bin/csh
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -t 0-08:00
#SBATCH -p general
#SBATCH --mem=60000
#SBATCH -o Logs/qpm_run.%A.out         # File to which STDOUT will be written
#SBATCH -e Logs/qpm_run.%A.err         # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=oliver.philcox@cfa.harvard.edu # Email to which notifications will be sent

# We could access a task ID by SLURM_ARRAY_TASK_ID

#setenv OMP_NUM_THREADS $SLURM_NPROCS
#set OMP_NUM_THREADS=32 ## for compatibility
#set SLURM_ARRAY_TASK_ID=22

set command = "./grid_multipolesAVX -box 1500 -scale 1 -rmax 180 -ngrid 40"
echo $command

set inn = /mnt/store1/oliverphilcox/GRFs_aper_dense_big2
set root = grf
set out = /mnt/store1/oliverphilcox/GRFs_aper_dense_big2/SE_out
mkdir $out

foreach mockno (`seq 0 2 299`)

    echo Using mock $mockno
    
    set outroot = $root.$mockno
    set multfile = $out/$root.$mockno.mult

    echo Running DDD counts
    $command -in $inn/mock_$mockno -save $multfile > $out/$outroot.ddd.out
    echo Done with DDD
    
    echo Starting RRR
    $command -in $inn/ran_0 -invert > $out/$outroot.rrr.out
    echo Done with RRR
    
    echo Starting NNN counts
    foreach n (`seq 0 1 12`)
        # Make new file of Data+Randoms
        cp -f $inn/mock_$mockno $inn/tmp_ran_$n.$mockno
        cat $inn/ran_$n >> $inn/tmp_ran_$n.$mockno
        
        echo Starting NNN_$n
        $command -in $inn/tmp_ran_$n.$mockno -load $multfile -balance > $out/$outroot.nnn_$n.out
        
        # Remove trash
        rm $inn/tmp_ran_$n.$mockno
    end
        
    rm $multfile
end

echo Computations Complete
#     
#     
# set root = qpm
# set ranroot = $root
# set outroot = $root.$mockno
# 
# mkdir $out
# rm -f $errlog
# date > $errlog
# echo Executing $0 >> $errlog
# echo Number of threads: $OMP_NUM_THREADS >> $errlog
# 
# 
# 
# 
# # Now make D-R and run that, with loading
# foreach n ( 00 01) #02 03 04 05 06 07 08 09 \
#         #10 11 12 13 14 15 16 17 18 19 \
# 	    #20 21 22 23 24 25 26 27 28 29 \
# 	    #30 31 32)
#     /bin/cp -f $tmp/$root.data $tmp/$root.ran.$n
#     gunzip -c $inn/$ranroot.ran.$n.gz >> $tmp/$root.ran.$n
# 
#     echo "Starting D-R $n" >> $errlog
#     date >> $errlog
#     ($command -in $tmp/$root.ran.$n -balance -load $multfile \
# 	    > $tmpout/$outroot.r$n.out) >>& $errlog
# 
#     /bin/rm -f $tmp/$root.ran.$n
#     echo Done with D-R $n
# 
# end    # foreach D-R loop
#  
# # Combine the files
# #module load hpc/python-2.7.3
# python2 ./sumfiles.py $tmpout $tmpout/$outroot.r??.out > $tmpout/$outroot.sum
# 
# echo Finished with computation.  Placing results into $out/$outroot.tgz
# echo Finished with computation.  Placing results into $out/$outroot.tgz >> $errlog
# date >> $errlog
# pushd $tmpout
# echo >> $errlog
# /bin/ls -l >> $errlog
# echo >> $errlog
# tail $outroot.sum >> $errlog
# /bin/cp $errlog .
# /bin/mv 3pcf_output.npz 3pcf_output_$mockno.npz
# tar cfz $outroot.tgz $outroot.*.out $outroot.sum $errfile 3pcf_output_$mockno.npz run_qpm.csh
# popd
# /bin/mv $tmpout/$outroot.tgz $tmpout/$outroot.sum $tmpout/3pcf_output_$mockno.npz $out
# 
# # Destroy ramdisk
# /bin/rm -rf $tmp
