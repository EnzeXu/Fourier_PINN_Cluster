#!/bin/bash
#SBATCH --job-name="zjx_sir_alpha_penalty"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/zjx_sir_alpha_penalty-%j.o"
#SBATCH --error="jobs_oe/zjx_sir_alpha_penalty-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python zjx_sir_alpha_penalty.py