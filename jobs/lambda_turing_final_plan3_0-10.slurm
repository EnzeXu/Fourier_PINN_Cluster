#!/bin/bash
#SBATCH --job-name="lambda_turing_final_plan3"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_turing_final_plan3-%j.o"
#SBATCH --error="jobs_oe/lambda_turing_final_plan3-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 0 --log_path logs/lambda_turing_final_plan3_0.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 1 --log_path logs/lambda_turing_final_plan3_1.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 2 --log_path logs/lambda_turing_final_plan3_2.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 3 --log_path logs/lambda_turing_final_plan3_3.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 4 --log_path logs/lambda_turing_final_plan3_4.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 5 --log_path logs/lambda_turing_final_plan3_5.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 6 --log_path logs/lambda_turing_final_plan3_6.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 7 --log_path logs/lambda_turing_final_plan3_7.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 8 --log_path logs/lambda_turing_final_plan3_8.txt
python model_Turing_Lambda.py --main_path . --layer 4 --init none --activation plan3 --seed 9 --log_path logs/lambda_turing_final_plan3_9.txt
