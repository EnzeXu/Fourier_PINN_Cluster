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

draft_head = """#!/bin/bash
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
"""

draft_normal = "python {0} {1}\n"

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
    full_path = "jobs/{}.slurm".format(job_name)
    print("build {}".format(full_path))
    with open(full_path, "w") as f:
        f.write(draft.format(
            job_name,
            python_name,
            " ".join(["--{0} {1}".format(one_key, kwargs[one_key]) for one_key in kwargs])
        ))

def one_slurm_multi_seed(job_name, python_name, kwargs, seed_start, seed_end, log_path_base, draft_head=draft_head, draft_normal=draft_normal):
    full_path = "jobs/{}_{}-{}.slurm".format(job_name, seed_start, seed_end)
    print("build {}".format(full_path))
    with open(full_path, "w") as f:
        f.write(draft_head.format(
            job_name
        ))
        for one_seed in range(seed_start, seed_end):
            kwargs["seed"] = one_seed
            kwargs["log_path"] = "logs/{}.txt".format(
                log_path_base.format(one_seed))
            f.write(draft_normal.format(
                python_name,
                " ".join(["--{0} {1}".format(one_key, kwargs[one_key]) for one_key in kwargs])
            ))

def one_time_build_pp_lambda():
    plans = [
        [0],
        [1],
        [2],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        for seed in range(2):
            dic["log_path"] = "logs/{}.txt".format(
                "lambda_pp_s{}_{}".format(one_plan[0], seed))
            dic["seed"] = seed
            dic["strategy"] = one_plan[0]
            one_slurm(
                "lambda_pp_s{}_{}".format(one_plan[0], seed),
                "model_PP_Lambda.py", dic)

def one_time_build_sir_lambda():
    plans = [
        [0],
        [1],
        [2],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        for seed in range(2):
            dic["log_path"] = "logs/{}.txt".format(
                "lambda_sir_s{}_{}".format(one_plan[0], seed))
            dic["seed"] = seed
            dic["strategy"] = one_plan[0]
            one_slurm(
                "lambda_sir_s{}_{}".format(one_plan[0], seed),
                "model_SIR_Lambda.py", dic)

def one_time_build_sir_lambda_final():
    plans = [
        ["plan3"],
        ["original"],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        # dic["seed"] = seed
        dic["activation"] = one_plan[0]
        one_slurm_multi_seed(
            "lambda_sir_final_{}".format(one_plan[0]),
            "model_SIR_Lambda.py", dic, 0, 10,
            "lambda_sir_final_{}_{{}}".format(one_plan[0]),
        )

def one_time_build_rep_lambda():
    plans = [
        [0],
        [1],
        [2],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        for seed in range(2):
            dic["log_path"] = "logs/{}.txt".format(
                "lambda_rep_s{}_{}".format(one_plan[0], seed))
            dic["seed"] = seed
            dic["strategy"] = one_plan[0]
            one_slurm(
                "lambda_rep_s{}_{}".format(one_plan[0], seed),
                "model_REP_Lambda.py", dic)

def one_time_build_rep_lambda_final():
    plans = [
        # activation / penalty / pinn
        ["plan3", 0, 1],
        ["plan3", 0, 0],
        ["plan3", 1, 0],
        ["original", 0, 0],
        ["original", 1, 0],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        dic["activation"] = one_plan[0]
        dic["penalty"] = one_plan[1]
        dic["pinn"] = one_plan[2]
        one_slurm_multi_seed(
            "lambda_rep_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_rep_final_pinn",
            "model_REP_Lambda.py", dic, 0, 10,
            "lambda_rep_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_rep_final_pinn_{}",
        )

def one_time_build_cc1_lambda():
    plans = [
        ["none", 0, 5],
        ["none", 5, 10],
        ["none", 10, 15],
        ["none", 15, 20],
        ["xavier_uniform", 0, 5],
        ["xavier_uniform", 5, 10],
        ["xavier_uniform", 10, 15],
        ["xavier_uniform", 15, 20],
        ["xavier_normal", 0, 5],
        ["xavier_normal", 5, 10],
        ["xavier_normal", 10, 15],
        ["xavier_normal", 15, 20],
        ["kaiming_uniform", 0, 5],
        ["kaiming_uniform", 5, 10],
        ["kaiming_uniform", 10, 15],
        ["kaiming_uniform", 15, 20],
        ["kaiming_normal", 0, 5],
        ["kaiming_normal", 5, 10],
        ["kaiming_normal", 10, 15],
        ["kaiming_normal", 15, 20],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        # dic["seed"] = seed
        dic["init"] = one_plan[0]
        one_slurm_multi_seed(
            "lambda_cc1_{}".format(one_plan[0]),
            "model_CC1_Lambda.py", dic, one_plan[1], one_plan[2],
            "lambda_cc1_{}_{{}}".format(one_plan[0]),
        )


def one_time_build_cc1_lambda_final():
    plans = [
        # activation / penalty / pinn
        ["plan3", 0, 1],
        ["plan3", 0, 0],
        ["plan3", 1, 0],
        ["original", 0, 0],
        ["original", 1, 0],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        dic["activation"] = one_plan[0]
        dic["penalty"] = one_plan[1]
        dic["pinn"] = one_plan[2]
        one_slurm_multi_seed(
            "lambda_cc1_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_cc1_final_pinn",
            "model_CC1_Lambda.py", dic, 0, 20,
            "lambda_cc1_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_cc1_final_pinn_{}",
        )

def one_time_build_turing_lambda_final():
    plans = [
        ["plan3"],
        ["original"],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        # dic["seed"] = seed
        # dic["init"] = "none"
        dic["activation"] = one_plan[0]
        one_slurm_multi_seed(
            "lambda_turing_final_{}".format(one_plan[0]),
            "model_Turing_Lambda.py", dic, 0, 10,
            "lambda_turing_final_{}_{{}}".format(one_plan[0]),
        )

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
                "model_SIR_Zeta.py", dic, draft)

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
                "model_REP_Zeta.py", dic, draft)

if __name__ == "__main__":
    # one_time_build_rep_zeta()
    # one_time_build_sir_zeta()
    # one_time_build_cc1_lambda()
    # one_time_build_sir_lambda_final()
    one_time_build_rep_lambda_final()
    # one_time_build_cc1_lambda_final()
    # one_time_build_turing_lambda_final()

    pass


