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


# draft_head = """#!/bin/bash
# #SBATCH --job-name="{0}"
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1
# #SBATCH --mem=8GB
# #SBATCH --ntasks-per-node=8
# #SBATCH --mail-user=xue20@wfu.edu
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output="jobs_oe/{0}-%j.o"
# #SBATCH --error="jobs_oe/{0}-%j.e"
#
# echo $(pwd) > "jobs/pwd.txt"
# source /deac/csc/chenGrp/software/tensorflow/bin/activate
# """

# #SBATCH --constraint=cascade
draft_head = """#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --partition=gpu
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

draft_head_cpu = """#!/bin/bash
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

def one_slurm_multi_seed(job_name, python_name, kwargs, seed_start, seed_end, log_path_base, draft_head=draft_head, draft_normal=draft_normal, cpu=False):
    full_path = "jobs/{}_{}-{}.slurm".format(job_name, seed_start, seed_end)
    draft_used = draft_head
    if cpu:
        draft_used = draft_head_cpu
    print("build {}".format(full_path))
    with open(full_path, "w") as f:
        f.write(draft_used.format(
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
            "lambda_sir_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_sir_final_pinn",
            "model_SIR_Lambda.py", dic, 0, 10,
            "lambda_sir_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_sir_final_pinn_{}",
        )

# def one_time_build_rep_lambda():
#     plans = [
#         [0],
#         [1],
#         [2],
#     ]
#     dic = dict()
#     dic["main_path"] = "."
#     dic["layer"] = 4
#     for one_plan in plans:
#         for seed in range(2):
#             dic["log_path"] = "logs/{}.txt".format(
#                 "lambda_rep_s{}_{}".format(one_plan[0], seed))
#             dic["seed"] = seed
#             dic["strategy"] = one_plan[0]
#             one_slurm(
#                 "lambda_rep_s{}_{}".format(one_plan[0], seed),
#                 "model_REP_Lambda.py", dic)

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

def one_time_build_toggle_lambda_final():
    plans = [
        # activation / penalty / pinn
        ["plan3", 0, 1],
        ["plan3", 0, 0],
        ["plan3", 1, 0],
        ["original", 0, 0],
        ["original", 1, 0],
    ]
    module_name_short = "Toggle"
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        dic["activation"] = one_plan[0]
        dic["penalty"] = one_plan[1]
        dic["pinn"] = one_plan[2]
        one_slurm_multi_seed(
            "lambda_{}_final_a={}_p={}".format(module_name_short.lower(), one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_{}_final_pinn".format(module_name_short.lower()),
            "model_{}_Lambda.py".format(module_name_short), dic, 0, 10,
            "lambda_{}_final_a={}_p={}_{{}}".format(module_name_short.lower(), one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_{}_final_pinn_{{}}".format(module_name_short.lower()),
            cpu=False,
        )

# def one_time_build_cc1_lambda():
#     plans = [
#         ["none", 0, 5],
#         ["none", 5, 10],
#         ["none", 10, 15],
#         ["none", 15, 20],
#         ["xavier_uniform", 0, 5],
#         ["xavier_uniform", 5, 10],
#         ["xavier_uniform", 10, 15],
#         ["xavier_uniform", 15, 20],
#         ["xavier_normal", 0, 5],
#         ["xavier_normal", 5, 10],
#         ["xavier_normal", 10, 15],
#         ["xavier_normal", 15, 20],
#         ["kaiming_uniform", 0, 5],
#         ["kaiming_uniform", 5, 10],
#         ["kaiming_uniform", 10, 15],
#         ["kaiming_uniform", 15, 20],
#         ["kaiming_normal", 0, 5],
#         ["kaiming_normal", 5, 10],
#         ["kaiming_normal", 10, 15],
#         ["kaiming_normal", 15, 20],
#     ]
#     dic = dict()
#     dic["main_path"] = "."
#     dic["layer"] = 4
#     for one_plan in plans:
#         # dic["seed"] = seed
#         dic["init"] = one_plan[0]
#         one_slurm_multi_seed(
#             "lambda_cc1_{}".format(one_plan[0]),
#             "model_CC1_Lambda.py", dic, one_plan[1], one_plan[2],
#             "lambda_cc1_{}_{{}}".format(one_plan[0]),
#         )


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
            "model_CC1_Lambda.py", dic, 0, 10,
            "lambda_cc1_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[2] else "lambda_cc1_final_pinn_{}",
        )
        one_slurm_multi_seed(
            "lambda_cc1_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_cc1_final_pinn",
            "model_CC1_Lambda.py", dic, 10, 20,
            "lambda_cc1_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_cc1_final_pinn_{}",
        )
        one_slurm_multi_seed(
            "lambda_cc1_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_cc1_final_pinn",
            "model_CC1_Lambda.py", dic, 20, 30,
            "lambda_cc1_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_cc1_final_pinn_{}",
        )
        one_slurm_multi_seed(
            "lambda_cc1_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_cc1_final_pinn",
            "model_CC1_Lambda.py", dic, 30, 40,
            "lambda_cc1_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_cc1_final_pinn_{}",
        )

def one_time_build_turing_lambda_final():
    plans = [
        # activation / penalty / pinn
        ["plan3", 0, 1],
        ["plan3", 0, 0],
        # ["plan3", 1, 0],
        ["original", 0, 0],
        # ["original", 1, 0],
    ]
    dic = dict()
    dic["main_path"] = "."
    dic["layer"] = 4
    for one_plan in plans:
        dic["activation"] = one_plan[0]
        dic["penalty"] = one_plan[1]
        dic["pinn"] = one_plan[2]
        one_slurm_multi_seed(
            "lambda_turing_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_turing_final_pinn",
            "model_Turing_Lambda.py", dic, 0, 10,
            "lambda_turing_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_turing_final_pinn_{}",
        )

def one_time_build_pp_lambda_final():
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
            "lambda_pp_final_a={}_p={}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_pp_final_pinn",
            "model_PP_Lambda.py", dic, 0, 10,
            "lambda_pp_final_a={}_p={}_{{}}".format(one_plan[0], one_plan[1]) if not one_plan[
                2] else "lambda_pp_final_pinn_{}",
        )

# def one_time_build_pp_zeta():
#     plans = [
#         ["plan1", -1],
#         ["plan2", 1],
#         ["plan2", 2],
#         ["plan2", 3],
#         ["plan2", 4],
#         ["plan2", 5],
#         ["plan3", -1],
#     ]
#     dic = dict()
#     dic["main_path"] = "."
#     dic["layer"] = 4
#     for one_plan in plans:
#         for seed in range(2):
#             dic["log_path"] = "logs/{}.txt".format(
#                 "zeta_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed))
#             dic["seed"] = seed
#             dic["activation"] = one_plan[0]
#             dic["activation_id"] = one_plan[1]
#             one_slurm(
#                 "zeta_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed),
#                 "model_PP_Zeta.py", dic)
#
# def one_time_build_sir_zeta():
#     plans = [
#         ["plan1", -1],
#         ["plan2", 1],
#         ["plan2", 2],
#         ["plan2", 3],
#         ["plan2", 4],
#         ["plan2", 5],
#         ["plan3", -1],
#     ]
#     dic = dict()
#     dic["main_path"] = "."
#     dic["layer"] = 4
#     for one_plan in plans:
#         for seed in range(2):
#             dic["log_path"] = "logs/{}.txt".format(
#                 "zeta_sir_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed))
#             dic["seed"] = seed
#             dic["activation"] = one_plan[0]
#             dic["activation_id"] = one_plan[1]
#             one_slurm(
#                 "zeta_sir_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed),
#                 "model_SIR_Zeta.py", dic, draft)
#
# def one_time_build_rep_zeta():
#     plans = [
#         ["plan1", -1],
#         ["plan2", 1],
#         ["plan2", 2],
#         ["plan2", 3],
#         ["plan2", 4],
#         ["plan2", 5],
#         ["plan3", -1],
#     ]
#     dic = dict()
#     dic["main_path"] = "."
#     dic["layer"] = 4
#     for one_plan in plans:
#         for seed in range(2):
#             dic["log_path"] = "logs/{}.txt".format(
#                 "zeta_rep_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed))
#             dic["seed"] = seed
#             dic["activation"] = one_plan[0]
#             dic["activation_id"] = one_plan[1]
#             one_slurm(
#                 "zeta_rep_{}{}_{}".format(one_plan[0], "-{}".format(one_plan[1]) if one_plan[0] == "plan2" else "", seed),
#                 "model_REP_Zeta.py", dic, draft)

def one_time_build_rep3_omega_activations():
    plans = [
        # pinn / activation / cyclic / stable / derivative / boundary
        [1, "adaptive", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 0],
        [0, "relu", 0, 0, 0, 0],
        [0, "elu", 0, 0, 0, 0],
        [0, "tanh", 0, 0, 0, 0],
        [0, "sin", 0, 0, 0, 0],
        [0, "softplus", 0, 0, 0, 0],
        [0, "adaptive", 0, 0, 0, 0],
        [0, "adaptive_3", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 1],
    ]
    module_name_short = "REP3"
    dic = dict()
    dic["main_path"] = "."
    dic["skip_draw_flag"] = 1
    for one_plan in plans:
        dic["pinn"] = one_plan[0]
        dic["activation"] = one_plan[1]
        dic["cyclic"] = one_plan[2]
        dic["stable"] = one_plan[3]
        dic["derivative"] = one_plan[4]
        dic["boundary"] = one_plan[5]

        if one_plan[0]:
            title_format = "o_{}_pinn".format(module_name_short.lower())
        elif one_plan[5]:
            title_format = "o_{}_boundary".format(module_name_short.lower())
        else:
            title_format = "o_{}_{}".format(module_name_short.lower(), one_plan[1])
        title_format_log = title_format + "_{}"

        one_slurm_multi_seed(
            title_format,
            "model_{}_Omega.py".format(module_name_short), dic, 2, 3,
            title_format_log,
            cpu=False,
        )

def one_time_build_rep6_omega_activations():
    plans = [
        # pinn / activation / cyclic / stable / derivative / boundary
        [1, "adaptive", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 0],
        [0, "relu", 0, 0, 0, 0],
        [0, "elu", 0, 0, 0, 0],
        [0, "tanh", 0, 0, 0, 0],
        [0, "sin", 0, 0, 0, 0],
        [0, "softplus", 0, 0, 0, 0],
        [0, "adaptive", 0, 0, 0, 0],
        [0, "adaptive_3", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 1],
    ]
    module_name_short = "REP6"
    dic = dict()
    dic["main_path"] = "."
    dic["skip_draw_flag"] = 1
    for one_plan in plans:
        dic["pinn"] = one_plan[0]
        dic["activation"] = one_plan[1]
        dic["cyclic"] = one_plan[2]
        dic["stable"] = one_plan[3]
        dic["derivative"] = one_plan[4]
        dic["boundary"] = one_plan[5]

        if one_plan[0]:
            title_format = "o_{}_pinn".format(module_name_short.lower())
        elif one_plan[5]:
            title_format = "o_{}_boundary".format(module_name_short.lower())
        else:
            title_format = "o_{}_{}".format(module_name_short.lower(), one_plan[1])
        title_format_log = title_format + "_{}"

        one_slurm_multi_seed(
            title_format,
            "model_{}_Omega.py".format(module_name_short), dic, 2, 3,
            title_format_log,
            cpu=False,
        )

def one_time_build_sir_omega_activations():
    plans = [
        # pinn / activation / cyclic / stable / derivative / boundary
        [1, "adaptive", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 0],
        [0, "relu", 0, 0, 0, 0],
        [0, "elu", 0, 0, 0, 0],
        [0, "tanh", 0, 0, 0, 0],
        [0, "sin", 0, 0, 0, 0],
        [0, "softplus", 0, 0, 0, 0],
        [0, "adaptive", 0, 0, 0, 0],
        [0, "adaptive_3", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 1],
    ]
    module_name_short = "SIR"
    dic = dict()
    dic["main_path"] = "."
    dic["skip_draw_flag"] = 1
    for one_plan in plans:
        dic["pinn"] = one_plan[0]
        dic["activation"] = one_plan[1]
        dic["cyclic"] = one_plan[2]
        dic["stable"] = one_plan[3]
        dic["derivative"] = one_plan[4]
        dic["boundary"] = one_plan[5]

        if one_plan[0]:
            title_format = "o_{}_pinn".format(module_name_short.lower())
        elif one_plan[5]:
            title_format = "o_{}_boundary".format(module_name_short.lower())
        else:
            title_format = "o_{}_{}".format(module_name_short.lower(), one_plan[1])
        title_format_log = title_format + "_{}"

        one_slurm_multi_seed(
            title_format,
            "model_{}_Omega.py".format(module_name_short), dic, 2, 3,
            title_format_log,
            cpu=False,
        )

def one_time_build_siraged_omega_activations():
    plans = [
        # pinn / activation / cyclic / stable / derivative / boundary
        [1, "adaptive", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 0],
        [0, "relu", 0, 0, 0, 0],
        [0, "elu", 0, 0, 0, 0],
        [0, "tanh", 0, 0, 0, 0],
        [0, "sin", 0, 0, 0, 0],
        [0, "softplus", 0, 0, 0, 0],
        [0, "adaptive", 0, 0, 0, 0],
        [0, "adaptive_3", 0, 0, 0, 0],
        [0, "gelu", 0, 0, 0, 1],
    ]
    module_name_short = "SIRAged"
    dic = dict()
    dic["main_path"] = "."
    dic["skip_draw_flag"] = 1
    for one_plan in plans:
        dic["pinn"] = one_plan[0]
        dic["activation"] = one_plan[1]
        dic["cyclic"] = one_plan[2]
        dic["stable"] = one_plan[3]
        dic["derivative"] = one_plan[4]
        dic["boundary"] = one_plan[5]

        if one_plan[0]:
            title_format = "o_{}_pinn".format(module_name_short.lower())
        elif one_plan[5]:
            title_format = "o_{}_boundary".format(module_name_short.lower())
        else:
            title_format = "o_{}_{}".format(module_name_short.lower(), one_plan[1])
        title_format_log = title_format + "_{}"

        one_slurm_multi_seed(
            title_format,
            "model_{}_Omega.py".format(module_name_short), dic, 2, 3,
            title_format_log,
            cpu=False,
        )

def one_time_build_omega(module_name_short, start_seed, end_seed):
    plans = [
        # pinn / activation / cyclic / stable / derivative / boundary
        # [1, "adaptive", 0, 0, 0, 0],
        # [0, "gelu", 0, 0, 0, 0],
        # [0, "relu", 0, 0, 0, 0],
        # [0, "elu", 0, 0, 0, 0],
        # [0, "tanh", 0, 0, 0, 0],
        # [0, "sin", 0, 0, 0, 0],
        # [0, "softplus", 0, 0, 0, 0],
        # [0, "adaptive", 0, 0, 0, 0],
        # [0, "adaptive_3", 0, 0, 0, 0],
        # [0, "gelu", 0, 0, 0, 1],
        # [0, "gelu", 1, 0, 0, 0],
        # [0, "gelu", 0, 0, 0, 2],
        [0, "gelu", 2, 0, 0, 0],
    ]
    module_name_short = module_name_short
    dic = dict()
    dic["main_path"] = "."
    dic["skip_draw_flag"] = 1
    for one_plan in plans:
        dic["pinn"] = one_plan[0]
        dic["activation"] = one_plan[1]
        dic["cyclic"] = one_plan[2]
        dic["stable"] = one_plan[3]
        dic["derivative"] = one_plan[4]
        dic["boundary"] = one_plan[5]

        if one_plan[0]:
            title_format = "o_{}_pinn".format(module_name_short.lower())
        elif one_plan[5]:
            title_format = "o_{}_boundary".format(module_name_short.lower())
        elif one_plan[2]:
            title_format = "o_{}_cyclic".format(module_name_short.lower())
        elif one_plan[3]:
            title_format = "o_{}_stable".format(module_name_short.lower())
        elif one_plan[4]:
            title_format = "o_{}_derivative".format(module_name_short.lower())
        else:
            title_format = "o_{}_{}".format(module_name_short.lower(), one_plan[1])
        title_format_log = title_format + "_{}"

        one_slurm_multi_seed(
            title_format,
            "model_{}_Omega.py".format(module_name_short), dic, start_seed, end_seed,
            title_format_log,
            cpu=False,
        )

if __name__ == "__main__":
    # one_time_build_rep_zeta()
    # one_time_build_sir_zeta()
    # one_time_build_cc1_lambda()
    # one_time_build_sir_lambda_final()
    # one_time_build_rep_lambda_final()
    # one_time_build_cc1_lambda_final()

    # one_time_build_omega("SIRAged", 0, 3)
    # one_time_build_omega("SIR", 0, 3)
    one_time_build_omega("REP6", 0, 3)
    one_time_build_omega("REP3", 0, 3)
    # one_time_build_rep3_omega_activations()
    # one_time_build_rep6_omega_activations()
    # one_time_build_sir_omega_activations()
    # one_time_build_siraged_omega_activations()
    # one_time_build_pp_lambda_final()
    # one_time_build_turing_lambda_final()

    pass


