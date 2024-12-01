#!/bin/bash

cd /scratch/09665/atokiwa/LSSC/nc1024_size625_tiled10

original_script="./jia-production-lensssc_temp.job"

seed=$1

temp_script="./temp_${seed}.job"

cp $original_script $temp_script

sed -i "s/seed/$seed/g" $temp_script

sbatch $temp_script

rm $temp_script