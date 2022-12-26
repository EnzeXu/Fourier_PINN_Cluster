#!/bin/bash
#SBATCH --job-name="lambda_turing_final_original"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_turing_final_original-%j.o"
#SBATCH --error="jobs_oe/lambda_turing_final_original-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 0 --log_path logs/lambda_turing_final_original_0.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 1 --log_path logs/lambda_turing_final_original_1.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 2 --log_path logs/lambda_turing_final_original_2.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 3 --log_path logs/lambda_turing_final_original_3.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 4 --log_path logs/lambda_turing_final_original_4.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 5 --log_path logs/lambda_turing_final_original_5.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 6 --log_path logs/lambda_turing_final_original_6.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 7 --log_path logs/lambda_turing_final_original_7.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 8 --log_path logs/lambda_turing_final_original_8.txt
python model_Turing_Lambda.py --main_path . --layer 4 --activation original --seed 9 --log_path logs/lambda_turing_final_original_9.txt
