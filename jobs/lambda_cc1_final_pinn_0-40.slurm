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
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 0 --log_path logs/lambda_cc1_final_pinn_0.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 1 --log_path logs/lambda_cc1_final_pinn_1.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 2 --log_path logs/lambda_cc1_final_pinn_2.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 3 --log_path logs/lambda_cc1_final_pinn_3.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 4 --log_path logs/lambda_cc1_final_pinn_4.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 5 --log_path logs/lambda_cc1_final_pinn_5.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 6 --log_path logs/lambda_cc1_final_pinn_6.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 7 --log_path logs/lambda_cc1_final_pinn_7.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 8 --log_path logs/lambda_cc1_final_pinn_8.txt
python model_CC1_Lambda.py --main_path . --layer 4 --activation plan3 --penalty 0 --pinn 1 --seed 9 --log_path logs/lambda_cc1_final_pinn_9.txt
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
