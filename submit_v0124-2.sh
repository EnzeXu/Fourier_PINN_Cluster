#!/bin/bash

for model_name in "rep3" "rep6" "siraged" "sir"
do
    sbatch jobs/o_${model_name}_pinn_0-2.slurm
    for activation in "adaptive" "gelu" "relu" "elu" "tanh" "sin" "softplus" "boundary"
    do
        sbatch jobs/o_${model_name}_${activation}_2-3.slurm
    done
done
