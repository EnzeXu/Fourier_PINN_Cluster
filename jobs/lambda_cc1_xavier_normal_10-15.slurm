#!/bin/bash
#SBATCH --job-name="lambda_cc1_xavier_normal"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_xavier_normal-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_xavier_normal-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --init xavier_normal --seed 10 --log_path logs/lambda_cc1_xavier_normal_10.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init xavier_normal --seed 11 --log_path logs/lambda_cc1_xavier_normal_11.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init xavier_normal --seed 12 --log_path logs/lambda_cc1_xavier_normal_12.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init xavier_normal --seed 13 --log_path logs/lambda_cc1_xavier_normal_13.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init xavier_normal --seed 14 --log_path logs/lambda_cc1_xavier_normal_14.txt
