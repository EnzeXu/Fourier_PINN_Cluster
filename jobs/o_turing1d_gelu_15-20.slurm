#!/bin/bash
#SBATCH --job-name="o_turing1d_gelu"
#SBATCH --partition=gpu
#SBATCH --constraint="skylake"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing1d_gelu-%j.o"
#SBATCH --error="jobs_oe/o_turing1d_gelu-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 15 --log_path logs/o_turing1d_gelu_15.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 16 --log_path logs/o_turing1d_gelu_16.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 17 --log_path logs/o_turing1d_gelu_17.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 18 --log_path logs/o_turing1d_gelu_18.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 19 --log_path logs/o_turing1d_gelu_19.txt
