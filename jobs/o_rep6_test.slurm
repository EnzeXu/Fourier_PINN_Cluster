#!/bin/bash
#SBATCH --job-name="o_rep6_test"
#SBATCH --partition=gpu
#SBATCH --constraint="skylake"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_rep6_test-%j.o"
#SBATCH --error="jobs_oe/o_rep6_test-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_REP6_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation gelu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.003 --init_weights avg --init_weights_strategy trainable --seed 0 --log_path logs/o_REP6_gelu_0.003_avg_trainable_0.txt --test 1
