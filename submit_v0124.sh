#!/bin/bash

for model_name in "rep6"
do
    sbatch jobs/o_${model_name}_pinn_0-2.slurm
    for activation in "adaptive" "gelu" "relu" "elu" "tanh" "sin" "softplus"
    do
        sbatch jobs/o_${model_name}_${activation}_0-2.slurm
    done
done
