#!/bin/bash
#SBATCH --job-name="zeta_rep_plan1_1"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/zeta_rep_plan1_1-%j.o"
#SBATCH --error="jobs_oe/zeta_rep_plan1_1-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python model_REP_Zeta.py --main_path . --layer 4 --log_path logs/zeta_rep_plan1_1.txt --seed 1 --activation plan1 --activation_id -1
