#!/bin/bash

for scheduler in "decade"
do
    for model_name in "turing1d" "turing2d"
    do
        for seed in "0-3"
        do
            for adaptive_activation in "adaptive_6" "adaptive_5"
            do
                for init_lr in "0.001" "0.003" "0.005" "0.01"
                do
                    sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${init_lr}_${seed}.slurm
                done
            done

            for adaptive_activation in "gelu" "relu" "elu" "tanh" "sin" "softplus"
            do
                sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${seed}.slurm
            done
        done
    done
done

