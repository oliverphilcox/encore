#!/bin/csh

@ i = 0
while ($i <= 33)
    set num = `printf %02d $i`
    echo $num
    mv /mnt/store1/oliverphilcox/3PCF_QPM/QPM/qpm.ran.$i.gz /mnt/store1/oliverphilcox/3PCF_QPM/QPM/qpm.ran.$num.gz
    @ i += 1
end
