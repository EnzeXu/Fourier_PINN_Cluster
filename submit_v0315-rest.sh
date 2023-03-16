#!/bin/bash

#for scheduler in "decade_pp"
#do
#    for model_name in "pp"
#    do
#        for seed in "0-1"
#        do
#            for adaptive_activation in "adaptive_6" "adaptive_5"
#            do
#                for init_lr in "0.001" "0.003" "0.005" "0.01"
#                do
#                    sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${init_lr}_${seed}.slurm
#                done
#            done
#
#            for adaptive_activation in "gelu" "relu" "elu" "tanh" "sin" "softplus"
#            do
#                sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${seed}.slurm
#            done
#        done
#    done
#done

for scheduler in "decade_pp"
do
    for model_name in "pp"
    do
        for seed in "1-3"
        do
            for adaptive_activation in "adaptive_6" "adaptive_5"
            do
                for init_lr in "0.001" "0.003" "0.005" "0.01"
                do
                    sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${init_lr}_boundary_${seed}.slurm
                done
            done

            for adaptive_activation in "gelu" "relu" "elu" "tanh" "sin" "softplus"
            do
                sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_boundary_${seed}.slurm
            done
        done
    done
done


for scheduler in "decade_pp"
do
    for model_name in "pp"
    do
        for seed in "3-5"
        do
            for adaptive_activation in "adaptive_6" "adaptive_5"
            do
                for init_lr in "0.001" "0.003" "0.005" "0.01"
                do
                    sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_${init_lr}_boundary_${seed}.slurm
                done
            done

            for adaptive_activation in "gelu" "relu" "elu" "tanh" "sin" "softplus"
            do
                sbatch jobs/o_${model_name}_${scheduler}_${adaptive_activation}_boundary_${seed}.slurm
            done
        done
    done
done
