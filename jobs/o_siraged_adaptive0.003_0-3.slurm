#!/bin/bash
#SBATCH --job-name="o_siraged_adaptive0.003"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_siraged_adaptive0.003-%j.o"
#SBATCH --error="jobs_oe/o_siraged_adaptive0.003-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 0 --log_path logs/o_siraged_adaptive0.003_0.txt --init_lr 0.003
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 1 --log_path logs/o_siraged_adaptive0.003_1.txt --init_lr 0.003
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 2 --log_path logs/o_siraged_adaptive0.003_2.txt --init_lr 0.003
