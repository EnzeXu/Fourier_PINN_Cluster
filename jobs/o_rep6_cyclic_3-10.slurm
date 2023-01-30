#!/bin/bash
#SBATCH --job-name="o_rep6_cyclic"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_rep6_cyclic-%j.o"
#SBATCH --error="jobs_oe/o_rep6_cyclic-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 1 --stable 0 --derivative 0 --boundary 0 --seed 3 --log_path logs/o_rep6_cyclic_3.txt
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 1 --stable 0 --derivative 0 --boundary 0 --seed 4 --log_path logs/o_rep6_cyclic_4.txt
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 1 --stable 0 --derivative 0 --boundary 0 --seed 5 --log_path logs/o_rep6_cyclic_5.txt
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 1 --stable 0 --derivative 0 --boundary 0 --seed 6 --log_path logs/o_rep6_cyclic_6.txt
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 1 --stable 0 --derivative 0 --boundary 0 --seed 7 --log_path logs/o_rep6_cyclic_7.txt
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 1 --stable 0 --derivative 0 --boundary 0 --seed 8 --log_path logs/o_rep6_cyclic_8.txt
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 1 --stable 0 --derivative 0 --boundary 0 --seed 9 --log_path logs/o_rep6_cyclic_9.txt
