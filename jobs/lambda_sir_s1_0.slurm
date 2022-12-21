#!/bin/bash
#SBATCH --job-name="lambda_sir_s1_0"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_sir_s1_0-%j.o"
#SBATCH --error="jobs_oe/lambda_sir_s1_0-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_SIR_Lambda.py --main_path . --layer 4 --log_path logs/lambda_sir_s1_0.txt --seed 0 --strategy 1
