#!/bin/bash
#SBATCH --job-name="o_turing1d_adaptive_5_0.005"
#SBATCH --partition=gpu
#SBATCH --constraint="cascade|skylake"
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing1d_adaptive_5_0.005-%j.o"
#SBATCH --error="jobs_oe/o_turing1d_adaptive_5_0.005-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.005 --seed 3 --log_path logs/o_turing1d_adaptive_5_0.005_3.txt
python model_Turing1D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.005 --seed 4 --log_path logs/o_turing1d_adaptive_5_0.005_4.txt
