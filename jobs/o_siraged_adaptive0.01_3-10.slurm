#!/bin/bash
#SBATCH --job-name="o_siraged_adaptive0.01"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_siraged_adaptive0.01-%j.o"
#SBATCH --error="jobs_oe/o_siraged_adaptive0.01-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 3 --log_path logs/o_siraged_adaptive0.01_3.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 4 --log_path logs/o_siraged_adaptive0.01_4.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 5 --log_path logs/o_siraged_adaptive0.01_5.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 6 --log_path logs/o_siraged_adaptive0.01_6.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 7 --log_path logs/o_siraged_adaptive0.01_7.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 8 --log_path logs/o_siraged_adaptive0.01_8.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.01 --seed 9 --log_path logs/o_siraged_adaptive0.01_9.txt
