#!/bin/bash

model_name="turing1d"

for seed in "5-10" "10-15" "15-20" "20-25" "25-30"
do
    for activation in "gelu" "elu"
    do
        sbatch jobs/o_${model_name}_${activation}_${seed}.slurm
    done
done
