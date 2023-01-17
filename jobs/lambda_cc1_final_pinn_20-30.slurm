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
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 20 --log_path logs/lambda_cc1_final_pinn_20.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 21 --log_path logs/lambda_cc1_final_pinn_21.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 22 --log_path logs/lambda_cc1_final_pinn_22.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 23 --log_path logs/lambda_cc1_final_pinn_23.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 24 --log_path logs/lambda_cc1_final_pinn_24.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 25 --log_path logs/lambda_cc1_final_pinn_25.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 26 --log_path logs/lambda_cc1_final_pinn_26.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 27 --log_path logs/lambda_cc1_final_pinn_27.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 28 --log_path logs/lambda_cc1_final_pinn_28.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 29 --log_path logs/lambda_cc1_final_pinn_29.txt
