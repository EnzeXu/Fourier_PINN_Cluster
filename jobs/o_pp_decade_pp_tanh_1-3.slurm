#!/bin/bash
#SBATCH --job-name="o_pp_decade_pp_tanh"
#SBATCH --partition=gpu
#SBATCH --constraint="skylake"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --account=chenGrp
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_pp_decade_pp_tanh-%j.o"
#SBATCH --error="jobs_oe/o_pp_decade_pp_tanh-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_PP_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation tanh --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --scheduler decade_pp --seed 1 --log_path logs/o_pp_decade_pp_tanh_1.txt
python model_PP_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation tanh --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --scheduler decade_pp --seed 2 --log_path logs/o_pp_decade_pp_tanh_2.txt
