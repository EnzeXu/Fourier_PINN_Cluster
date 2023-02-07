#!/bin/bash
#SBATCH --job-name="o_turing2d_adaptive_0.001"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing2d_adaptive_0.001-%j.o"
#SBATCH --error="jobs_oe/o_turing2d_adaptive_0.001-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 0 --log_path logs/o_turing2d_adaptive_0.001_0.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 1 --log_path logs/o_turing2d_adaptive_0.001_1.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 2 --log_path logs/o_turing2d_adaptive_0.001_2.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 3 --log_path logs/o_turing2d_adaptive_0.001_3.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --seed 4 --log_path logs/o_turing2d_adaptive_0.001_4.txt
