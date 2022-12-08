import pickle
import numpy as np
import json
import torch

from utils import *
from model_PP_Eta import Config, Parameters, TrainArgs

results_info_dict = [
    ["PP_Fourier_Alpha", "alpha (seed=0)", "20221206_183712", "r"],  # 20221205_002014
    ["PP_Fourier_Alpha", "alpha (seed=1)", "20221206_205113", "r"],  # 20221205_013253
    # ["PP_Fourier_Beta", "beta (seed=0)", "20221205_001134", "b"],
    # ["PP_Fourier_Beta", "beta (seed=1)", "20221205_013403", "b"],
    # ["PP_Fourier_Delta", "delta (seed=0)", "20221204_203244", "g"],
    # ["PP_Fourier_Delta", "delta (seed=1)", "20221204_191307", "g"],
    # ["PP_Fourier_Eta", "eta (seed=0)", "20221205_072329", "orange"],
    # ["PP_Fourier_Eta", "eta (seed=1)", "20221205_083354", "orange"],
    ["PP_Fourier_Epsilon1", "epsilon1 (seed=0)", "20221206_054012", "g"],  # 20221206_054012
    ["PP_Fourier_Epsilon1", "epsilon1 (seed=1)", "20221206_073304", "g"],  # 20221206_073304
    ["PP_Fourier_Epsilon2", "epsilon2 (seed=0)", "20221206_054243", "b"],
    ["PP_Fourier_Epsilon2", "epsilon2 (seed=1)", "20221206_074715", "b"],
    ["PP_Fourier_Epsilon3", "epsilon3 (seed=0)", "20221206_184712", "orange"],
    ["PP_Fourier_Epsilon3", "epsilon3 (seed=1)", "20221206_203739", "orange"],
]

def read_one(path):
    # with open(path, "rb") as f:
    #     data = pickle.load(f)
    data = torch.load(path, map_location=torch.device("cpu"))
    print(json.dumps(data, indent=4))


def smooth_conv(data, kernel_size: int = 10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')


def one_time_draw_results_structured(info_dict, diff_start_list=[0, 10000, 100000, 450000], diff_end_list=[499000, 499000, 499000, 499000]):
    n_info = len(results_info_dict)
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

    m = MultiSubplotDraw(row=1, col=n_diff_start, fig_size=(16 * n_diff_start, 9), tight_layout_flag=False)
    for one_start, one_end in zip(diff_start_list, diff_end_list):
        ax = m.add_subplot(
            y_lists=[smooth_conv(data_list[i]["real_loss"], kernel_size=10)[one_start: one_end] for i in range(n_info)],
            x_list=range(max_x)[one_start:],
            color_list=color_list,
            legend_list=["{0} (end at loss = {1:.6e})".format(label_list[i], sum(data_list[i]["real_loss"][-10:]) / 10) for i in range(n_info)],
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

    m = MultiSubplotDraw(row=1, col=n_diff_start, fig_size=(16 * n_diff_start, 9), tight_layout_flag=False)
    for one_start, one_end in zip(diff_start_list, diff_end_list):
        ax = m.add_subplot(
            y_lists=[smooth_conv(data_list[i]["loss"], kernel_size=1000)[one_start: one_end] for i in range(n_info)],
            x_list=range(max_x)[one_start:],
            color_list=color_list,
            legend_list=["{0} (end at loss = {1:.6e})".format(label_list[i], sum(data_list[i]["real_loss"][-10:]) / 10) for i in range(n_info)],
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
    read_one("saves/train/PP_Fourier_Eta_20221207_152440/PP_Fourier_Eta_20221207_152440_info.npy")
    pass
