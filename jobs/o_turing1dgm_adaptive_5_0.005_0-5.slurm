#!/bin/bash
#SBATCH --job-name="o_turing1dgm_adaptive_5_0.005"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/o_turing1dgm_adaptive_5_0.005-%j.o"
#SBATCH --error="jobs_oe/o_turing1dgm_adaptive_5_0.005-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python model_Turing1DGM_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.005 --seed 0 --log_path logs/o_turing1dgm_adaptive_5_0.005_0.txt
python model_Turing1DGM_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.005 --seed 1 --log_path logs/o_turing1dgm_adaptive_5_0.005_1.txt
python model_Turing1DGM_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.005 --seed 2 --log_path logs/o_turing1dgm_adaptive_5_0.005_2.txt
python model_Turing1DGM_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.005 --seed 3 --log_path logs/o_turing1dgm_adaptive_5_0.005_3.txt
python model_Turing1DGM_Omega.py --main_path . --skip_draw_flag 1 --pinn 0 --activation adaptive_5 --cyclic 0 --stable 0 --derivative 0 --boundary 0 --init_lr 0.005 --seed 4 --log_path logs/o_turing1dgm_adaptive_5_0.005_4.txt
