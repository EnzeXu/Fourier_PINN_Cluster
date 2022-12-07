#!/bin/bash -l

#SBATCH --job-name="eta_32_0"
#SBATCH --partition=gpu
#SBATCH --nodes=1

#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/eta_32_0-%j.o"
#SBATCH --error="jobs_oe/eta_32_0-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_PP_Eta.py --log_path logs/eta_32_0.txt --main_path . --layer 32 --seed 0