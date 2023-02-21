#!/bin/bash

model_name="turing2d"

for seed in "0-1" "1-5"
do
    sbatch jobs/o_${model_name}_pinn_${seed}.slurm

    for activation in "gelu" "relu" "elu" "tanh" "sin" "softplus" "boundary" "stable"
    do
        sbatch jobs/o_${model_name}_${activation}_${seed}.slurm
    done

    for adaptive_activation in "adaptive_6" "adaptive_5"
    do
        for init_lr in "0.001" "0.003" "0.005" "0.01"
        do
            sbatch jobs/o_${model_name}_${adaptive_activation}_${init_lr}_avg_trainable_${seed}.slurm
        done
    done
done
