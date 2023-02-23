#!/bin/bash

for model_name in "sir" "siraged" "rep3" "rep6"
do
    for seed in "0-5"
    do
#        sbatch jobs/o_${model_name}_pinn_${seed}.slurm

        for adaptive_activation in "adaptive_6" "adaptive_5"
        do
            for init_lr in "0.001" "0.003" "0.005" "0.01"
            do
                sbatch jobs/o_${model_name}_${adaptive_activation}_${init_lr}_avg_trainable_${seed}.slurm
            done
        done
    done
done
