#!/bin/bash
#SBATCH --job-name="lambda_cc1_kaiming_uniform"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_kaiming_uniform-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_kaiming_uniform-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 5 --log_path logs/lambda_cc1_kaiming_uniform_5.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 6 --log_path logs/lambda_cc1_kaiming_uniform_6.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 7 --log_path logs/lambda_cc1_kaiming_uniform_7.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 8 --log_path logs/lambda_cc1_kaiming_uniform_8.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 9 --log_path logs/lambda_cc1_kaiming_uniform_9.txt
