#!/bin/bash
#SBATCH --job-name="lambda_cc1_final_plan3"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/lambda_cc1_final_plan3-%j.o"
#SBATCH --error="jobs_oe/lambda_cc1_final_plan3-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 0 --log_path logs/lambda_cc1_final_plan3_0.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 1 --log_path logs/lambda_cc1_final_plan3_1.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 2 --log_path logs/lambda_cc1_final_plan3_2.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 3 --log_path logs/lambda_cc1_final_plan3_3.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 4 --log_path logs/lambda_cc1_final_plan3_4.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 5 --log_path logs/lambda_cc1_final_plan3_5.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 6 --log_path logs/lambda_cc1_final_plan3_6.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 7 --log_path logs/lambda_cc1_final_plan3_7.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 8 --log_path logs/lambda_cc1_final_plan3_8.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 9 --log_path logs/lambda_cc1_final_plan3_9.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 10 --log_path logs/lambda_cc1_final_plan3_10.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 11 --log_path logs/lambda_cc1_final_plan3_11.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 12 --log_path logs/lambda_cc1_final_plan3_12.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 13 --log_path logs/lambda_cc1_final_plan3_13.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 14 --log_path logs/lambda_cc1_final_plan3_14.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 15 --log_path logs/lambda_cc1_final_plan3_15.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 16 --log_path logs/lambda_cc1_final_plan3_16.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 17 --log_path logs/lambda_cc1_final_plan3_17.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 18 --log_path logs/lambda_cc1_final_plan3_18.txt
python model_CC1_Lambda.py --main_path . --layer 4 --init none --seed 19 --log_path logs/lambda_cc1_final_plan3_19.txt
