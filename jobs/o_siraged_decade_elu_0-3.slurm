#!/bin/bash
#SBATCH --job-name="o_siraged_decade_elu"
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
#SBATCH --output="jobs_oe/o_siraged_decade_elu-%j.o"
#SBATCH --error="jobs_oe/o_siraged_decade_elu-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --scheduler decade --seed 0 --log_path logs/o_siraged_decade_elu_0.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --scheduler decade --seed 1 --log_path logs/o_siraged_decade_elu_1.txt
python model_SIRAged_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation elu --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.001 --scheduler decade --seed 2 --log_path logs/o_siraged_decade_elu_2.txt
