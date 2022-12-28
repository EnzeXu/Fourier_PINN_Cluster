#!/bin/bash
sbatch jobs/lambda_turing_final_a\=original_p\=0_0-10.slurm
sbatch jobs/lambda_turing_final_a\=plan3_p\=0_0-10.slurm

sbatch jobs/lambda_cc1_final_a\=original_p\=0_0-40.slurm
sbatch jobs/lambda_cc1_final_a\=original_p\=1_0-40.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=0_0-40.slurm
sbatch jobs/lambda_cc1_final_a\=plan3_p\=1_0-40.slurm

sbatch jobs/lambda_rep_final_a\=original_p\=0_0-10.slurm
sbatch jobs/lambda_rep_final_a\=original_p\=1_0-10.slurm
sbatch jobs/lambda_rep_final_a\=plan3_p\=0_0-10.slurm
sbatch jobs/lambda_rep_final_a\=plan3_p\=1_0-10.slurm

sbatch jobs/lambda_sir_final_a\=original_p\=0_0-10.slurm
sbatch jobs/lambda_sir_final_a\=plan3_p\=0_0-10.slurm