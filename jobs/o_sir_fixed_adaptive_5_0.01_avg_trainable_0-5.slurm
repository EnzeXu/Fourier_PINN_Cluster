#!/bin/bash
#SBATCH --job-name="o_sir_fixed_adaptive_5_0.01_avg_trainable"
#SBATCH --partition=gpu
#SBATCH --constraint="skylake"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_sir_fixed_adaptive_5_0.01_avg_trainable-%j.o"
#SBATCH --error="jobs_oe/o_sir_fixed_adaptive_5_0.01_avg_trainable-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_SIR_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --scheduler fixed --init_weights avg --init_weights_strategy trainable --seed 0 --log_path logs/o_sir_fixed_adaptive_5_0.01_avg_trainable_0.txt
python model_SIR_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --scheduler fixed --init_weights avg --init_weights_strategy trainable --seed 1 --log_path logs/o_sir_fixed_adaptive_5_0.01_avg_trainable_1.txt
python model_SIR_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --scheduler fixed --init_weights avg --init_weights_strategy trainable --seed 2 --log_path logs/o_sir_fixed_adaptive_5_0.01_avg_trainable_2.txt
python model_SIR_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --scheduler fixed --init_weights avg --init_weights_strategy trainable --seed 3 --log_path logs/o_sir_fixed_adaptive_5_0.01_avg_trainable_3.txt
python model_SIR_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --scheduler fixed --init_weights avg --init_weights_strategy trainable --seed 4 --log_path logs/o_sir_fixed_adaptive_5_0.01_avg_trainable_4.txt
