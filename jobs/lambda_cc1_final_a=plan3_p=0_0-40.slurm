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
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 30 --log_path logs/lambda_cc1_final_a=plan3_p=0_30.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 31 --log_path logs/lambda_cc1_final_a=plan3_p=0_31.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 32 --log_path logs/lambda_cc1_final_a=plan3_p=0_32.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 33 --log_path logs/lambda_cc1_final_a=plan3_p=0_33.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 34 --log_path logs/lambda_cc1_final_a=plan3_p=0_34.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 35 --log_path logs/lambda_cc1_final_a=plan3_p=0_35.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 36 --log_path logs/lambda_cc1_final_a=plan3_p=0_36.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 37 --log_path logs/lambda_cc1_final_a=plan3_p=0_37.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 38 --log_path logs/lambda_cc1_final_a=plan3_p=0_38.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 0 --seed 39 --log_path logs/lambda_cc1_final_a=plan3_p=0_39.txt
