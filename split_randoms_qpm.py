## Code to split the QPM DR12 randoms into 1.5x N_gal chunks

import numpy as np

ranfile = '/mnt/store1/oliverphilcox/PowerSpectra/qpm_randoms_50x.xyzwj'
galfile = '/mnt/store1/oliverphilcox/PowerSpectra/qpm_galaxy_1.xyzwj'
outdir = '/mnt/store1/oliverphilcox/3PCF_QPM/'

# Count number of galaxies
with open(galfile,"r") as in_gal:
    for l,line in enumerate(in_gal):
        pass;
N_gal = l+1
print("N_gal = %d"%N_gal)
N_chunk = int(N_gal*1.5)
print("N_rand_chunk = %d"%N_chunk)

with open(ranfile,"r") as in_ran:
    for l,line in enumerate(in_ran):
        if l%N_chunk==0:
            this_ind = l//N_chunk
            outfile = outdir+'qpm.ran.%d'%this_ind
            print("Writing to file %s"%outfile);
            out_ran = open(outfile,"w+")
        split_line=np.array(line.strip().split(" "), dtype=float) 
        x=split_line[0];
        y=split_line[1];
        z=split_line[2];
        w=split_line[3];
        out_ran.write("%.2f %.2f %.2f %.2f\n"%(x,y,z,-w))

print("Process complete");

