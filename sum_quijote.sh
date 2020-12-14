outdir=/tigress/ophilcox/quijote_3pcf/slepian_output_100
module load anaconda3
source activate py2

for mockno in {0..101}
do
echo $mockno
mkdir $outdir/quijote_$mockno
touch tmp
python sumfiles.py $outdir/quijote_$mockno $outdir/quijote.$mockno.r??.out > tmp
rm tmp
done
