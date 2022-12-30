#!/bin/bash
#SBATCH --job-name="lambda_pp_final_pinn"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_pp_final_pinn-%j.o"
#SBATCH --error="jobs_oe/lambda_pp_final_pinn-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 0 --log_path logs/lambda_pp_final_pinn_0.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 1 --log_path logs/lambda_pp_final_pinn_1.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 2 --log_path logs/lambda_pp_final_pinn_2.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 3 --log_path logs/lambda_pp_final_pinn_3.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 4 --log_path logs/lambda_pp_final_pinn_4.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 5 --log_path logs/lambda_pp_final_pinn_5.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 6 --log_path logs/lambda_pp_final_pinn_6.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 7 --log_path logs/lambda_pp_final_pinn_7.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 8 --log_path logs/lambda_pp_final_pinn_8.txt
python model_PP_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 9 --log_path logs/lambda_pp_final_pinn_9.txt
