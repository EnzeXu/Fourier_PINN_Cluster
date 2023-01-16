#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_a=plan3_p=0"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_a=plan3_p=0-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_a=plan3_p=0-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 20 --log_path logs/lambda_cc1_final_a=plan3_p=0_20.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 21 --log_path logs/lambda_cc1_final_a=plan3_p=0_21.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 22 --log_path logs/lambda_cc1_final_a=plan3_p=0_22.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 23 --log_path logs/lambda_cc1_final_a=plan3_p=0_23.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 24 --log_path logs/lambda_cc1_final_a=plan3_p=0_24.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 25 --log_path logs/lambda_cc1_final_a=plan3_p=0_25.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 26 --log_path logs/lambda_cc1_final_a=plan3_p=0_26.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 27 --log_path logs/lambda_cc1_final_a=plan3_p=0_27.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 28 --log_path logs/lambda_cc1_final_a=plan3_p=0_28.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 29 --log_path logs/lambda_cc1_final_a=plan3_p=0_29.txt
