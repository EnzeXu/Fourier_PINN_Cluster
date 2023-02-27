#!/bin/bash

for model_name in "sir" "siraged" "rep3" "rep6"
do
    for seed in "0-5"
    do
#        sbatch jobs/o_${model_name}_pinn_${seed}.slurm

        for adaptive_activation in "gelu" "relu" "elu" "tanh" "sin" "softplus"
        do
            sbatch jobs/o_${model_name}_${adaptive_activation}_${seed}.slurm
        done
    done
done