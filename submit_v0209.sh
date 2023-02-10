#!/bin/bash

for model_name in "turing1d" "turing2d"
do
    sbatch jobs/o_${model_name}_pinn_0-3.slurm
    for activation in "gelu" "relu" "elu" "tanh" "sin" "softplus" "boundary" "stable" "adaptive_0.001" "adaptive_0.003" "adaptive_0.005" "adaptive_0.01" "adaptive_5_0.001" "adaptive_5_0.003" "adaptive_5_0.005" "adaptive_5_0.01"
    do
        sbatch jobs/o_${model_name}_${activation}_0-3.slurm
    done
done
