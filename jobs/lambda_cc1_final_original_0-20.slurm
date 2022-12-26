#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_original"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_original-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_original-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 0 --log_path logs/lambda_cc1_final_original_0.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 1 --log_path logs/lambda_cc1_final_original_1.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 2 --log_path logs/lambda_cc1_final_original_2.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 3 --log_path logs/lambda_cc1_final_original_3.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 4 --log_path logs/lambda_cc1_final_original_4.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 5 --log_path logs/lambda_cc1_final_original_5.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 6 --log_path logs/lambda_cc1_final_original_6.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 7 --log_path logs/lambda_cc1_final_original_7.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 8 --log_path logs/lambda_cc1_final_original_8.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 9 --log_path logs/lambda_cc1_final_original_9.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 10 --log_path logs/lambda_cc1_final_original_10.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 11 --log_path logs/lambda_cc1_final_original_11.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 12 --log_path logs/lambda_cc1_final_original_12.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 13 --log_path logs/lambda_cc1_final_original_13.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 14 --log_path logs/lambda_cc1_final_original_14.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 15 --log_path logs/lambda_cc1_final_original_15.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 16 --log_path logs/lambda_cc1_final_original_16.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 17 --log_path logs/lambda_cc1_final_original_17.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 18 --log_path logs/lambda_cc1_final_original_18.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --activation original --seed 19 --log_path logs/lambda_cc1_final_original_19.txt
