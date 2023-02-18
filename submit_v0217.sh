#!/bin/bash

for model_name in "turing1d"
do
    for activation in "adaptive_2_0.01"
    do
        for weights in "avg" "gelu" "softplus"
        do
            for strategy in "trainable" "fixed"
            do
                sbatch jobs/o_${model_name}_${activation}_${weights}_${strategy}_0-1.slurm
            done
        done
    done
done

for model_name in "turing1d"
do
    for activation in "adaptive_6_0.01"
    do
        for weights in "avg" "gelu" "softplus" "relu" "elu" "tanh" "sin"
        do
            for strategy in "trainable" "fixed"
            do
                sbatch jobs/o_${model_name}_${activation}_${weights}_${strategy}_0-1.slurm
            done
        done
    done
done
