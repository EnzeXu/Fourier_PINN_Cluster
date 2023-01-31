#!/bin/bash

for model_name in "rep3" "rep6" "siraged" "sir"
do
    for activation in "adaptive_5_0.001" "adaptive_5_0.003" "adaptive_5_0.005" "adaptive_5_0.01"
    do
        sbatch jobs/o_${model_name}_${activation}_0-5.slurm
    done
done



