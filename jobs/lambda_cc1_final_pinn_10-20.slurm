#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_pinn"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_pinn-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_pinn-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 10 --log_path logs/lambda_cc1_final_pinn_10.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 11 --log_path logs/lambda_cc1_final_pinn_11.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 12 --log_path logs/lambda_cc1_final_pinn_12.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 13 --log_path logs/lambda_cc1_final_pinn_13.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 14 --log_path logs/lambda_cc1_final_pinn_14.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 15 --log_path logs/lambda_cc1_final_pinn_15.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 16 --log_path logs/lambda_cc1_final_pinn_16.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 17 --log_path logs/lambda_cc1_final_pinn_17.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 18 --log_path logs/lambda_cc1_final_pinn_18.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 19 --log_path logs/lambda_cc1_final_pinn_19.txt
