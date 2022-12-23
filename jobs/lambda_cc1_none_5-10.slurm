#!/bin/bash
#SBATCH --job-name="lambda_cc1_none"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_none-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_none-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 5 --log_path logs/lambda_cc1_none_5.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 6 --log_path logs/lambda_cc1_none_6.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 7 --log_path logs/lambda_cc1_none_7.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 8 --log_path logs/lambda_cc1_none_8.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 9 --log_path logs/lambda_cc1_none_9.txt
