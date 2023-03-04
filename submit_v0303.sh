#!/bin/bash

for scheduler in "fixed" "cosine" "decade"
do
    for model_name in "turing1d" "turing2d" "sir" "siraged" "rep6" "rep3"
    do
        for seed in "0-5"
        do
            for adaptive_activation in "adaptive_6" "adaptive_5"
            do
                for init_lr in "0.001" "0.003" "0.005" "0.01"
                do
                    sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${init_lr}_avg_trainable_${seed}.slurm
                done
            done
        done
    done
done

