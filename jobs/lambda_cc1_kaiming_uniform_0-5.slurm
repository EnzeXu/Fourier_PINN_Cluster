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
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 0 --log_path logs/lambda_cc1_kaiming_uniform_0.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 1 --log_path logs/lambda_cc1_kaiming_uniform_1.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 2 --log_path logs/lambda_cc1_kaiming_uniform_2.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 3 --log_path logs/lambda_cc1_kaiming_uniform_3.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init kaiming_uniform --seed 4 --log_path logs/lambda_cc1_kaiming_uniform_4.txt
