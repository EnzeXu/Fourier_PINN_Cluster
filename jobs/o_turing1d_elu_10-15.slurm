#!/bin/bash
#SBATCH --job-name="o_turing1d_elu"
#SBATCH --partition=gpu
#SBATCH --constraint="skylake"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing1d_elu-%j.o"
#SBATCH --error="jobs_oe/o_turing1d_elu-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 10 --log_path logs/o_turing1d_elu_10.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 11 --log_path logs/o_turing1d_elu_11.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 12 --log_path logs/o_turing1d_elu_12.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 13 --log_path logs/o_turing1d_elu_13.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 14 --log_path logs/o_turing1d_elu_14.txt
