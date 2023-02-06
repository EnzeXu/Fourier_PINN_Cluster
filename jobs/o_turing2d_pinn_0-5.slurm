#!/bin/bash
#SBATCH --job-name="o_turing2d_pinn"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing2d_pinn-%j.o"
#SBATCH --error="jobs_oe/o_turing2d_pinn-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 1 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 0 --log_path logs/o_turing2d_pinn_0.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 1 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 1 --log_path logs/o_turing2d_pinn_1.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 1 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 2 --log_path logs/o_turing2d_pinn_2.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 1 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 3 --log_path logs/o_turing2d_pinn_3.txt
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 1 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 4 --log_path logs/o_turing2d_pinn_4.txt
