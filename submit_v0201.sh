#!/bin/bash

for model_name in "rep3" "rep6" "siraged" "sir"
do
    for activation in "boundary"
    do
        sbatch jobs/o_${model_name}_${activation}_0-10.slurm
    done
done



