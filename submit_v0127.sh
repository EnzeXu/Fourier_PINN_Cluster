#!/bin/bash

for model_name in "rep3" "rep6" "siraged" "sir"
do
    for activation in "adaptive0.001" "adaptive0.003" "adaptive0.005" "adaptive0.01"
    do
        sbatch jobs/o_${model_name}_${activation}_0-3.slurm
    done
done

for model_name in "rep3" "rep6" "siraged" "sir"
do
    for activation in "gelu" "elu" "softplus" "sin" "tanh" "relu"
    do
        sbatch jobs/o_${model_name}_${activation}_0-3.slurm
    done
done

for model_name in "rep3" "rep6"
do
    for activation in "boundary" "cyclic"
    do
        sbatch jobs/o_${model_name}_${activation}_0-3.slurm
    done
done

for model_name in "siraged" "sir"
do
    for activation in "boundary" "stable"
    do
        sbatch jobs/o_${model_name}_${activation}_0-3.slurm
    done
done

for model_name in "rep3" "rep6" "siraged" "sir"
do
    sbatch jobs/o_${model_name}_pinn_0-3.slurm
done
