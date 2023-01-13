#!/bin/bash

sbatch jobs/lambda_turing_final_pinn_0-10.slurm
sbatch jobs/lambda_turing_final_a\=original_p\=0_0-10.slurm
sbatch jobs/lambda_turing_final_a\=plan3_p\=0_0-10.slurm

sbatch jobs/lambda_sir_final_pinn_0-10.slurm
sbatch jobs/lambda_sir_final_a\=original_p\=0_0-10.slurm
sbatch jobs/lambda_sir_final_a\=plan3_p\=0_0-10.slurm

sbatch jobs/lambda_rep_final_pinn_0-10.slurm
sbatch jobs/lambda_rep_final_a\=original_p\=0_0-10.slurm
sbatch jobs/lambda_rep_final_a\=original_p\=1_0-10.slurm
sbatch jobs/lambda_rep_final_a\=plan3_p\=0_0-10.slurm
sbatch jobs/lambda_rep_final_a\=plan3_p\=1_0-10.slurm

#sbatch jobs/lambda_sir_final_pinn_0-10.slurm
#sbatch jobs/lambda_sir_final_a\=original_p\=0_0-10.slurm
#sbatch jobs/lambda_sir_final_a\=plan3_p\=0_0-10.slurm

sbatch jobs/lambda_cc1_final_a\=original_p\=0_0-10.slurm
sbatch jobs/lambda_cc1_final_a\=original_p\=0_10-20.slurm
sbatch jobs/lambda_cc1_final_a\=original_p\=0_20-30.slurm
sbatch jobs/lambda_cc1_final_a\=original_p\=0_30-40.slurm

sbatch jobs/lambda_cc1_final_a\=original_p\=1_0-10.slurm
sbatch jobs/lambda_cc1_final_a\=original_p\=1_10-20.slurm
sbatch jobs/lambda_cc1_final_a\=original_p\=1_20-30.slurm
sbatch jobs/lambda_cc1_final_a\=original_p\=1_30-40.slurm

sbatch jobs/lambda_cc1_final_a\=plan3_p\=0_0-10.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=0_10-20.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=0_20-30.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=0_30-40.slurm

sbatch jobs/lambda_cc1_final_a\=plan3_p\=1_0-10.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=1_10-20.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=1_20-30.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=1_30-40.slurm

sbatch jobs/lambda_cc1_final_pinn_0-10.slurm
sbatch jobs/lambda_cc1_final_pinn_10-20.slurm
sbatch jobs/lambda_cc1_final_pinn_20-30.slurm
sbatch jobs/lambda_cc1_final_pinn_30-40.slurm