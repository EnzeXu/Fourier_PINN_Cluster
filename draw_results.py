import pickle
import numpy as np
import json
import torch

from utils import *
from model_REP_Zeta import Config, Parameters, TrainArgs

results_info_dict_zjx = [
    ["PP_Fourier_Alpha", "plan1 (seed=0)", "20221208_174052", "r"],
    ["REP_Fourier_Zeta", "plan1 (seed=1)", "20221208_174542", "r"],
    ["REP_Fourier_Zeta", "plan2-1 (seed=0)", "20221208_174724", "g"],
    ["REP_Fourier_Zeta", "plan2-1 (seed=1)", "20221208_175655", "g"],
    ["REP_Fourier_Zeta", "plan2-2 (seed=0)", "20221208_180424", "b"],
    ["REP_Fourier_Zeta", "plan2-2 (seed=1)", "20221208_181031", "b"],
    ["REP_Fourier_Zeta", "plan2-3 (seed=0)", "20221208_181226", "cyan"],
    ["REP_Fourier_Zeta", "plan2-3 (seed=1)", "20221208_182228", "cyan"],
    ["REP_Fourier_Zeta", "plan2-4 (seed=0)", "20221208_182743", "orange"],
    ["REP_Fourier_Zeta", "plan2-4 (seed=1)", "20221208_183324", "orange"],
    ["REP_Fourier_Zeta", "plan2-5 (seed=0)", "20221208_183647", "pink"],
    ["REP_Fourier_Zeta", "plan2-5 (seed=1)", "20221208_184819", "pink"],
]

results_info_dict_REP = [
    ["zjx_rep_alpha_penalty", "alpha", "20221215_013835", "lime"],
    ["REP_Fourier_Lambda", "plan3 - scale=ones (seed=0)", "20221214_220113", "r"],
    ["REP_Fourier_Lambda", "plan3 - scale=ones (seed=1)", "20221214_220713", "r"],
    # # ["REP_Fourier_Lambda", "plan3 - scale=fixed (seed=0)", "20221214_234356", "g"],
    # ["REP_Fourier_Lambda", "plan3 - scale=fixed (seed=1)", "20221214_234739", "g"],
    # # ["REP_Fourier_Lambda", "plan3 - scale=adaptive (seed=0)", "20221215_000135", "b"],
    # ["REP_Fourier_Lambda", "plan3 - scale=adaptive (seed=1)", "20221215_001208", "b"],
]

# results_info_dict_REP = [
#     ["REP_Fourier_Zeta", "plan1 (seed=0)", "20221208_174052", "r"],
#     ["REP_Fourier_Zeta", "plan1 (seed=1)", "20221208_174542", "r"],
#     ["REP_Fourier_Zeta", "plan2-1 (seed=0)", "20221208_174724", "g"],
#     ["REP_Fourier_Zeta", "plan2-1 (seed=1)", "20221208_175655", "g"],
#     ["REP_Fourier_Zeta", "plan2-2 (seed=0)", "20221208_180424", "b"],
#     ["REP_Fourier_Zeta", "plan2-2 (seed=1)", "20221208_181031", "b"],
#     ["REP_Fourier_Zeta", "plan2-3 (seed=0)", "20221208_181226", "cyan"],
#     ["REP_Fourier_Zeta", "plan2-3 (seed=1)", "20221208_182228", "cyan"],
#     ["REP_Fourier_Zeta", "plan2-4 (seed=0)", "20221208_182743", "orange"],
#     ["REP_Fourier_Zeta", "plan2-4 (seed=1)", "20221208_183324", "orange"],
#     ["REP_Fourier_Zeta", "plan2-5 (seed=0)", "20221208_183647", "pink"],
#     ["REP_Fourier_Zeta", "plan2-5 (seed=1)", "20221208_184819", "pink"],
#     ["REP_Fourier_Zeta", "plan3 (seed=0)", "20221213_144957", "lime"],
#     ["REP_Fourier_Zeta", "plan3 (seed=1)", "20221213_145036", "lime"],
# ]

results_info_dict_PP = [
    ["zjx_alpha_penalty", "alpha", "20221215_013100", "lime"],
    ["zjx_alpha_penalty", "alpha", "20221215_025825", "lime"],
    ["zjx_alpha_penalty", "alpha", "20221215_025832", "lime"],
    ["zjx_alpha_penalty", "alpha", "20221215_035158", "lime"],
    ["PP_Fourier_Lambda", "plan3 - scale=ones (seed=0)", "20221214_150347", "r"],
    ["PP_Fourier_Lambda", "plan3 - scale=ones (seed=1)", "20221214_150352", "r"],
    # ["PP_Fourier_Lambda", "plan3 - scale=fixed (seed=0)", "20221214_155843", "g"],
    # ["PP_Fourier_Lambda", "plan3 - scale=fixed (seed=1)", "20221214_173034", "g"],
    # ["PP_Fourier_Lambda", "plan3 - scale=adaptive (seed=0)", "20221214_191147", "b"],
    # ["PP_Fourier_Lambda", "plan3 - scale=adaptive (seed=1)", "20221214_192603", "b"],
    # ["PP_Fourier_Lambda", "plan3 - scale=adaptive & sin.omega fixed (seed=0)", "20221215_023915", "orange"],
    # ["PP_Fourier_Lambda", "plan3 - scale=adaptive & sin.omega fixed (seed=1)", "20221215_023921", "orange"],
]

# results_info_dict_PP = [
#     ["PP_Fourier_Zeta", "plan1 (seed=0)", "20221208_141621", "r"],
#     ["PP_Fourier_Zeta", "plan1 (seed=1)", "20221208_141624", "r"],
#     ["PP_Fourier_Zeta", "plan2-1 (seed=0)", "20221208_142156", "g"],
#     ["PP_Fourier_Zeta", "plan2-1 (seed=1)", "20221208_142159", "g"],
#     ["PP_Fourier_Zeta", "plan2-2 (seed=0)", "20221208_155517", "b"],
#     ["PP_Fourier_Zeta", "plan2-2 (seed=1)", "20221208_155548", "b"],
#     ["PP_Fourier_Zeta", "plan2-3 (seed=0)", "20221208_155819", "cyan"],
#     ["PP_Fourier_Zeta", "plan2-3 (seed=1)", "20221208_160220", "cyan"],
#     ["PP_Fourier_Zeta", "plan2-4 (seed=0)", "20221208_160406", "orange"],
#     ["PP_Fourier_Zeta", "plan2-4 (seed=1)", "20221208_160814", "orange"],
#     ["PP_Fourier_Zeta", "plan2-5 (seed=0)", "20221208_173344", "pink"],
#     ["PP_Fourier_Zeta", "plan2-5 (seed=1)", "20221208_173420", "pink"],
#     ["PP_Fourier_Zeta", "plan3 (seed=0)", "20221213_143030", "lime"],
#     ["PP_Fourier_Zeta", "plan3 (seed=1)", "20221213_143032", "lime"],
#     ["zjx_alpha", "alpha", "20221213_050019", "brown"],
#     ["zjx_w", "w", "20221213_050022", "tan"],
#     ["zjx_combine", "combine", "20221213_050025", "gold"],
# ]
results_info_dict_SIR = [
    ["zjx_sir_alpha_penalty", "alpha", "20221215_013529", "lime"],
    ["SIR_Fourier_Lambda", "plan3 - scale=ones (seed=0)", "20221214_200815", "r"],
    ["SIR_Fourier_Lambda", "plan3 - scale=ones (seed=1)", "20221214_202254", "r"],
    # ["SIR_Fourier_Lambda", "plan3 - scale=fixed (seed=0)", "20221214_203729", "g"],
    # ["SIR_Fourier_Lambda", "plan3 - scale=fixed (seed=1)", "20221214_205208", "g"],
    # ["SIR_Fourier_Lambda", "plan3 - scale=adaptive (seed=0)", "20221214_210643", "b"],
    # ["SIR_Fourier_Lambda", "plan3 - scale=adaptive (seed=1)", "20221214_212241", "b"],
]
# results_info_dict_SIR = [
#     ["SIR_Fourier_Zeta", "plan1 (seed=0)", "20221208_185112", "r"],
#     ["SIR_Fourier_Zeta", "plan1 (seed=1)", "20221208_185310", "r"],
#     ["SIR_Fourier_Zeta", "plan2-1 (seed=0)", "20221208_185601", "g"],
#     ["SIR_Fourier_Zeta", "plan2-1 (seed=1)", "20221208_185519", "g"],
#     ["SIR_Fourier_Zeta", "plan2-2 (seed=0)", "20221208_185712", "b"],
#     ["SIR_Fourier_Zeta", "plan2-2 (seed=1)", "20221208_185750", "b"],
#     ["SIR_Fourier_Zeta", "plan2-3 (seed=0)", "20221208_185903", "cyan"],
#     ["SIR_Fourier_Zeta", "plan2-3 (seed=1)", "20221208_185943", "cyan"],
#     ["SIR_Fourier_Zeta", "plan2-4 (seed=0)", "20221208_190058", "orange"],
#     ["SIR_Fourier_Zeta", "plan2-4 (seed=1)", "20221208_190124", "orange"],
#     ["SIR_Fourier_Zeta", "plan2-5 (seed=0)", "20221208_190135", "pink"],
#     ["SIR_Fourier_Zeta", "plan2-5 (seed=1)", "20221208_190248", "pink"],
#     ["SIR_Fourier_Zeta", "plan3 (seed=0)", "20221213_144751", "lime"],
#     ["SIR_Fourier_Zeta", "plan3 (seed=1)", "20221213_145031", "lime"],
# ]

def read_one(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # data = torch.load(path, map_location=torch.device("cpu"))
    print(json.dumps(data, indent=4))


def smooth_conv(data, kernel_size: int = 10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')


def one_time_draw_results_structured(info_dict, save_path, diff_start_list, diff_end_list, ks=100):
    n_info = len(info_dict)
    model_name_list = [item[0] for item in info_dict]
    label_list = [item[1] for item in info_dict]
    time_string_list = [item[2] for item in info_dict]
    color_list = [item[3] for item in info_dict]
    data_list = []
    for one_model_name, one_time_string in zip(model_name_list, time_string_list):
        with open("saves/train/{0}_{1}/{0}_{1}_info.npy".format(one_model_name, one_time_string), "rb") as f:
            data_list.append(pickle.load(f))
    max_x = max([len(data_list[i]["real_loss"]) for i in range(n_info)])
    n_diff_start = len(diff_start_list)

    m = MultiSubplotDraw(row=1, col=n_diff_start, fig_size=(16 * n_diff_start, 9), tight_layout_flag=False, save_flag=True, save_path=save_path[0])
    for one_start, one_end in zip(diff_start_list, diff_end_list):
        ax = m.add_subplot(
            y_lists=[smooth_conv(data_list[i]["real_loss"], kernel_size=ks)[one_start: one_end] for i in range(n_info)],
            x_list=range(max_x)[one_start:],
            color_list=color_list,
            legend_list=["{0} (end at loss = {1:.6e})".format(label_list[i], sum(smooth_conv(data_list[i]["real_loss"], kernel_size=ks)[-10:]) / 10) for i in range(n_info)],
            line_style_list=["solid"] * n_info,
            x_label_size=25,
            y_label_size=25,
            fig_x_label="Epoch",
            fig_y_label="Real Loss",
            legend_fontsize=15,
            line_width=2,
        )
    plt.suptitle("Real Loss (MSE loss)", fontsize=25)
    m.draw()

    m = MultiSubplotDraw(row=1, col=n_diff_start, fig_size=(16 * n_diff_start, 9), tight_layout_flag=False, save_flag=True, save_path=save_path[1])
    for one_start, one_end in zip(diff_start_list, diff_end_list):
        ax = m.add_subplot(
            y_lists=[smooth_conv(data_list[i]["loss"], kernel_size=ks)[one_start: one_end] for i in range(n_info)],
            x_list=range(max_x)[one_start:],
            color_list=color_list,
            legend_list=["{0} (end at loss = {1:.6e})".format(label_list[i], sum(smooth_conv(data_list[i]["loss"], kernel_size=ks)[-10:]) / 10) for i in range(n_info)],
            line_style_list=["solid"] * n_info,
            x_label_size=25,
            y_label_size=25,
            fig_x_label="Epoch",
            fig_y_label="Training Loss",
            legend_fontsize=15,
            line_width=2,
        )
    plt.suptitle("Training Loss", fontsize=25)
    m.draw()

# one_time_draw_results_structured(results_info_dict)


if __name__ == "__main__":
    # read_one("saves/train/PP_Fourier_Eta_20221207_152440/PP_Fourier_Eta_20221207_152440_info.npy")
    print("helloworld!")
    one_time_draw_results_structured(results_info_dict_PP, ["test/test_pp_real_scale.png", "test/test_pp_scale.png"], [0, 100000, 290000], [299000, 299000, 299000], ks=1000)
    one_time_draw_results_structured(results_info_dict_REP, ["test/test_rep_real_scale.png", "test/test_rep_train_scale.png"], [0, 10000, 100000], [105000, 105000, 105000], ks=3000)
    one_time_draw_results_structured(results_info_dict_SIR, ["test/test_sir_real_scale.png", "test/test_sir_train_scale.png"], [0, 2000, 15000], [16000, 16000, 16000], ks=2000)
    pass
