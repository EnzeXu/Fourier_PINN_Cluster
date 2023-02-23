#!/bin/bash

model_name="turing1d"

for seed in "5-10" "10-15" "15-20" "20-25" "25-30"
do
    for adaptive_activation in "adaptive_6" "adaptive_5"
    do
        for init_lr in "0.001" "0.003" "0.005"
        do
            sbatch jobs/o_${model_name}_${adaptive_activation}_${init_lr}_avg_trainable_${seed}.slurm
        done
    done
done
