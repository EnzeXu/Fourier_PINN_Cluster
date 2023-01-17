#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_a=plan3_p=1"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_a=plan3_p=1-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_a=plan3_p=1-%j.e"
#
# echo $(pwd) > "jobs/pwd.txt"
# source /deac/csc/chenGrp/software/tensorflow/bin/activate
# python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 0 --log_path logs/lambda_cc1_final_a=plan3_p=1_0.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 1 --log_path logs/lambda_cc1_final_a=plan3_p=1_1.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 2 --log_path logs/lambda_cc1_final_a=plan3_p=1_2.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 3 --log_path logs/lambda_cc1_final_a=plan3_p=1_3.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 4 --log_path logs/lambda_cc1_final_a=plan3_p=1_4.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 5 --log_path logs/lambda_cc1_final_a=plan3_p=1_5.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 6 --log_path logs/lambda_cc1_final_a=plan3_p=1_6.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 7 --log_path logs/lambda_cc1_final_a=plan3_p=1_7.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 8 --log_path logs/lambda_cc1_final_a=plan3_p=1_8.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 1 --pinn 0 --seed 9 --log_path logs/lambda_cc1_final_a=plan3_p=1_9.txt
