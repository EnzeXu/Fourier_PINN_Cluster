draft = """#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --partition=gpu
#SBATCH --constraint=cascade
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/{0}-%j.o"
#SBATCH --error="jobs_oe/{0}-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python {1} {2}
"""

draft_cpu = """#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/{0}-%j.o"
#SBATCH --error="jobs_oe/{0}-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python {1} {2}
"""


def one_slurm(job_name, python_name, kwargs, draft=draft):
    with open("jobs/{}.slurm".format(job_name), "w") as f:
        f.write(draft.format(
            job_name,
            python_name,
            " ".join(["--{0} {1}".format(one_key, kwargs[one_key]) for one_key in kwargs])
        ))


def one_time_build_pp_zeta():
    plans = [
        ["plan1", -1],
        ["plan2", 1],
        ["plan2", 2],
        ["plan2", 3],
        ["plan2", 4],
        ["plan2", 5],
        ["plan3", -1],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        for seed in range(2):
            dic["log_path"] = "logs/{}.txt".format(
                "zeta_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed))
            dic["seed"] = seed
            dic["activation"] = one_plan[0]
            dic["activation_id"] = one_plan[1]
            one_slurm(
                "zeta_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed),
                "model_PP_Zeta.py", dic)

def one_time_build_sir_zeta():
    plans = [
        ["plan1", -1],
        ["plan2", 1],
        ["plan2", 2],
        ["plan2", 3],
        ["plan2", 4],
        ["plan2", 5],
        ["plan3", -1],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        for seed in range(2):
            dic["log_path"] = "logs/{}.txt".format(
                "zeta_sir_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed))
            dic["seed"] = seed
            dic["activation"] = one_plan[0]
            dic["activation_id"] = one_plan[1]
            one_slurm(
                "zeta_sir_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed),
                "model_SIR_Zeta.py", dic, draft_cpu)

def one_time_build_rep_zeta():
    plans = [
        ["plan1", -1],
        ["plan2", 1],
        ["plan2", 2],
        ["plan2", 3],
        ["plan2", 4],
        ["plan2", 5],
        ["plan3", -1],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        for seed in range(2):
            dic["log_path"] = "logs/{}.txt".format(
                "zeta_rep_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed))
            dic["seed"] = seed
            dic["activation"] = one_plan[0]
            dic["activation_id"] = one_plan[1]
            one_slurm(
                "zeta_rep_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed),
                "model_REP_Zeta.py", dic, draft_cpu)

if __name__ == "__main__":
    one_time_build_rep_zeta()
    pass
