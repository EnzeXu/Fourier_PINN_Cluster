#!/bin/bash
#SBATCH --job-name="o_rep3_adaptive_5_0.003"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_rep3_adaptive_5_0.003-%j.o"
#SBATCH --error="jobs_oe/o_rep3_adaptive_5_0.003-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_REP3_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.003 --seed 0 --log_path logs/o_rep3_adaptive_5_0.003_0.txt
python model_REP3_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.003 --seed 1 --log_path logs/o_rep3_adaptive_5_0.003_1.txt
python model_REP3_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.003 --seed 2 --log_path logs/o_rep3_adaptive_5_0.003_2.txt
python model_REP3_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.003 --seed 3 --log_path logs/o_rep3_adaptive_5_0.003_3.txt
python model_REP3_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.003 --seed 4 --log_path logs/o_rep3_adaptive_5_0.003_4.txt
