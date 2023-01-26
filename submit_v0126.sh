#!/bin/bash

for model_name in "rep3" "rep6" "siraged" "sir"
do
    for activation in "adaptive" "adaptive_3"
    do
        sbatch jobs/o_${model_name}_${activation}_0-2.slurm
    done
done

for model_name in "rep3" "rep6" "siraged" "sir"
do
    for activation in "adaptive" "adaptive_3"
    do
        sbatch jobs/o_${model_name}_${activation}_2-3.slurm
    done
done
