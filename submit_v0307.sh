#!/bin/bash

for scheduler in "fixed" "cosine" "decade"
do
    for model_name in "turing1d" "turing2d"
    do
        for seed in "0-5"
        do
            for adaptive_activation in "gelu" "relu" "elu" "tanh" "sin" "softplus"
            do
                sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${seed}.slurm
            done
        done
    done
done

