#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_pinn"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_pinn-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_pinn-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 30 --log_path logs/lambda_cc1_final_pinn_30.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 31 --log_path logs/lambda_cc1_final_pinn_31.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 32 --log_path logs/lambda_cc1_final_pinn_32.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 33 --log_path logs/lambda_cc1_final_pinn_33.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 34 --log_path logs/lambda_cc1_final_pinn_34.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 35 --log_path logs/lambda_cc1_final_pinn_35.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 36 --log_path logs/lambda_cc1_final_pinn_36.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 37 --log_path logs/lambda_cc1_final_pinn_37.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 38 --log_path logs/lambda_cc1_final_pinn_38.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 39 --log_path logs/lambda_cc1_final_pinn_39.txt
