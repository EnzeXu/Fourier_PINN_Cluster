#!/bin/bash
#SBATCH --job-name="o_turing2d_adaptive_6_0.01_avg_trainable"
#SBATCH --partition=gpu
#SBATCH --constraint="skylake"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing2d_adaptive_6_0.01_avg_trainable-%j.o"
#SBATCH --error="jobs_oe/o_turing2d_adaptive_6_0.01_avg_trainable-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing2D_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_6 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --init_weights avg --init_weights_strategy trainable --seed 0 --log_path logs/o_turing2d_adaptive_6_0.01_avg_trainable_0.txt
