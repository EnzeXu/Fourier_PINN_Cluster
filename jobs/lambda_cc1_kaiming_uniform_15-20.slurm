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
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 15 --log_path logs/lambda_cc1_kaiming_uniform_15.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 16 --log_path logs/lambda_cc1_kaiming_uniform_16.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 17 --log_path logs/lambda_cc1_kaiming_uniform_17.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 18 --log_path logs/lambda_cc1_kaiming_uniform_18.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 19 --log_path logs/lambda_cc1_kaiming_uniform_19.txt
