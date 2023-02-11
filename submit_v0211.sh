#!/bin/bash

for model_name in "turing1d"
do
    sbatch jobs/o_${model_name}_pinn_0-5.slurm
    for activation in "gelu" "relu" "elu" "tanh" "sin" "softplus" "boundary" "stable"
    do
        sbatch jobs/o_${model_name}_${activation}_0-5.slurm
    done
done
