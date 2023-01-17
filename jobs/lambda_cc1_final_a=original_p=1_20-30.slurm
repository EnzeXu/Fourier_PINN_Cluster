#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_a=original_p=1"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_a=original_p=1-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_a=original_p=1-%j.e"
#
# echo $(pwd) > "jobs/pwd.txt"
# source /deac/csc/chenGrp/software/tensorflow/bin/activate
# python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 20 --log_path logs/lambda_cc1_final_a=original_p=1_20.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 21 --log_path logs/lambda_cc1_final_a=original_p=1_21.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 22 --log_path logs/lambda_cc1_final_a=original_p=1_22.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 23 --log_path logs/lambda_cc1_final_a=original_p=1_23.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 24 --log_path logs/lambda_cc1_final_a=original_p=1_24.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 25 --log_path logs/lambda_cc1_final_a=original_p=1_25.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 26 --log_path logs/lambda_cc1_final_a=original_p=1_26.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 27 --log_path logs/lambda_cc1_final_a=original_p=1_27.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 28 --log_path logs/lambda_cc1_final_a=original_p=1_28.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation original --penalty 1 --pinn 0 --seed 29 --log_path logs/lambda_cc1_final_a=original_p=1_29.txt
