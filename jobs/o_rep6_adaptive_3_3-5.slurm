#!/bin/bash
#SBATCH --job-name="o_rep6_adaptive_3"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_rep6_adaptive_3-%j.o"
#SBATCH --error="jobs_oe/o_rep6_adaptive_3-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_3 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 3 --log_path logs/o_rep6_adaptive_3_3.txt
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_3 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --seed 4 --log_path logs/o_rep6_adaptive_3_4.txt
