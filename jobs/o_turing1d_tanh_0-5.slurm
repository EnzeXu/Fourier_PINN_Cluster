#!/bin/bash
#SBATCH --job-name="o_turing1d_tanh"
#SBATCH --partition=gpu
#SBATCH --constraint="cascade"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing1d_tanh-%j.o"
#SBATCH --error="jobs_oe/o_turing1d_tanh-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation tanh --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 0 --log_path logs/o_turing1d_tanh_0.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation tanh --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 1 --log_path logs/o_turing1d_tanh_1.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation tanh --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 2 --log_path logs/o_turing1d_tanh_2.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation tanh --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 3 --log_path logs/o_turing1d_tanh_3.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation tanh --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 4 --log_path logs/o_turing1d_tanh_4.txt
