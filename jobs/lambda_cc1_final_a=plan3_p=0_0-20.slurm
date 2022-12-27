#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_a=plan3_p=0"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_a=plan3_p=0-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_a=plan3_p=0-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 0 --log_path logs/lambda_cc1_final_a=plan3_p=0_0.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 1 --log_path logs/lambda_cc1_final_a=plan3_p=0_1.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 2 --log_path logs/lambda_cc1_final_a=plan3_p=0_2.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 3 --log_path logs/lambda_cc1_final_a=plan3_p=0_3.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 4 --log_path logs/lambda_cc1_final_a=plan3_p=0_4.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 5 --log_path logs/lambda_cc1_final_a=plan3_p=0_5.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 6 --log_path logs/lambda_cc1_final_a=plan3_p=0_6.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 7 --log_path logs/lambda_cc1_final_a=plan3_p=0_7.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 8 --log_path logs/lambda_cc1_final_a=plan3_p=0_8.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 9 --log_path logs/lambda_cc1_final_a=plan3_p=0_9.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 10 --log_path logs/lambda_cc1_final_a=plan3_p=0_10.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 11 --log_path logs/lambda_cc1_final_a=plan3_p=0_11.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 12 --log_path logs/lambda_cc1_final_a=plan3_p=0_12.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 13 --log_path logs/lambda_cc1_final_a=plan3_p=0_13.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 14 --log_path logs/lambda_cc1_final_a=plan3_p=0_14.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 15 --log_path logs/lambda_cc1_final_a=plan3_p=0_15.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 16 --log_path logs/lambda_cc1_final_a=plan3_p=0_16.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 17 --log_path logs/lambda_cc1_final_a=plan3_p=0_17.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 18 --log_path logs/lambda_cc1_final_a=plan3_p=0_18.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 19 --log_path logs/lambda_cc1_final_a=plan3_p=0_19.txt
