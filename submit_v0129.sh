#!/bin/bash

for model_name in "rep3" "rep6"
do
    for activation in "boundary" "cyclic"
    do
        sbatch jobs/o_${model_name}_${activation}_0-3.slurm
    done
done

for model_name in "siraged" "sir"
do
    for activation in "boundary" "stable"
    do
        sbatch jobs/o_${model_name}_${activation}_0-3.slurm
    done
done

