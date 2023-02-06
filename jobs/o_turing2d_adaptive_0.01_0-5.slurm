#!/bin/bash
#SBATCH --job-name="o_turing2d_adaptive_0.01"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing2d_adaptive_0.01-%j.o"
#SBATCH --error="jobs_oe/o_turing2d_adaptive_0.01-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 0 --log_path logs/o_turing2d_adaptive_0.01_0.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 1 --log_path logs/o_turing2d_adaptive_0.01_1.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 2 --log_path logs/o_turing2d_adaptive_0.01_2.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 3 --log_path logs/o_turing2d_adaptive_0.01_3.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 4 --log_path logs/o_turing2d_adaptive_0.01_4.txt
