#!/bin/bash
#SBATCH --job-name="omega_test"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/omega_test-%j.o"
#SBATCH --error="jobs_oe/omega_test-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_REP6_Omega.py --log_path logs/test.txt --main_path . --seed 0 --activation gelu --cyclic 0 --stable 0 --derivative 0 --skip_draw_flag 1 --pinn 1