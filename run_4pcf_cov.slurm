#!/bin/csh

#SBATCH -n 16 # cpus
#SBATCH -N 1 # tasks
#SBATCH -t 0-02:59:59 # time
#SBATCH --mem-per-cpu=1GB
#SBATCH -o /home/ophilcox/out/boss4pcfSall_run.%A.out         # File to which STDOUT will be written (make sure the directory exists!)
#SBATCH -e /home/ophilcox/out/boss4pcfSall_run.%A.err         # File to which STDERR will be written
#SBATCH --mail-type=END,FAIL        # Type of email notification
#SBATCH --mail-user=ophilcox@princeton.edu # Email to which notifications will be sent

##################### INPUT PARAMETERS ###################

for iell in {40..91}; do
  current_date_time="`date +%Y%m%d:%H:%M:%S`"
  echo $current_date_time
  idx_min=$(($iell-1))
  echo "ell_min=" $idx_min "ell_max=" $iell
  python wrapper_4pcf_cov.py $idx_min $iell &
done
#wait
current_date_time="`date +%Y%m%d:%H:%M:%S`"
