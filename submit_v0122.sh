#!/bin/bash

for model_name in "rep3" "rep6"
do
    sbatch jobs/o_${model_name}_pinn_0-10.slurm
    for activation in "gelu" "relu" "elu" "tanh" "sin" "softplus"
    do
        sbatch jobs/o_${model_name}_${activation}_0-10.slurm
    done
done
