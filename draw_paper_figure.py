import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils import smooth_conv


class MyNumpy:
    def __init__(self):
        pass

    @staticmethod
    def mean_array(x, skip: int = 0):
        x = x.flatten()
        assert len(x) > 2 * skip
        x_ordered = sorted(x)
        return np.sum(x_ordered[skip: -skip]) / (len(x) - 2 * skip)

    @staticmethod
    def max_array(x, skip: int = 0):
        x = x.flatten()
        assert len(x) > skip
        x_ordered = sorted(x)
        return x_ordered[- skip - 1]

    @staticmethod
    def min_array(x, skip: int = 0):
        x = x.flatten()
        assert len(x) > skip
        x_ordered = sorted(x)
        return x_ordered[skip]

    @staticmethod
    def mean(x, skip=0):
        assert x.ndim == 2
        x_mean = [MyNumpy.mean_array(x[:, i], skip) for i in range(x.shape[1])]
        return np.asarray(x_mean)

    @staticmethod
    def max(x, skip=0):
        assert x.ndim == 2
        x_mean = [MyNumpy.max_array(x[:, i], skip) for i in range(x.shape[1])]
        return np.asarray(x_mean)

    @staticmethod
    def min(x, skip=0):
        assert x.ndim == 2
        x_mean = [MyNumpy.min_array(x[:, i], skip) for i in range(x.shape[1])]
        return np.asarray(x_mean)



str1 = """
20221229_194306
20221229_200333
20221229_200838
20221229_193822
20221229_201045
20221229_194304
20221229_195307
20221229_195809
"""

str2 = """
20221229_210920
20221229_211653
20221229_203141
20221229_213310
20221229_210828
20221229_211653
20221229_213401
20221229_205915
20221229_210736
"""

str3 = """
20221229_194348
20221229_200515
20221229_201017
20221229_200808
20221229_203313
20221229_204315
20221229_204815
"""

str4 = """
20221229_215821
20221229_220554
20221229_212101
20221229_222128
20221229_221744
20221229_222524
20221229_215615
20221229_221134
20221229_221912
"""

def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def clear_reformat(string):
    lines = string.split()
    lines = ["\"{}\"".format(item) for item in lines if len(item) > 2]
    res_string = "[{}],".format(", ".join(lines))
    print(res_string)



def draw_paper_figure_loss(**kwargs):
    assert_keyword_list = ["timestring_dict", "info_path_format_dict", "model_name_short", "kernel_size", "mask_gap", "epoch_max", "y_ticks", "ylim", "y_ticks_format"]
    assert all(item in kwargs for item in assert_keyword_list)
    timestring_dict = kwargs["timestring_dict"]
    info_path_format_dict = kwargs["info_path_format_dict"]
    model_name_short = kwargs["model_name_short"]
    kernel_size = kwargs["kernel_size"]
    mask_gap = kwargs["mask_gap"]
    epoch_max = kwargs["epoch_max"]
    y_ticks = kwargs["y_ticks"]
    ylim = kwargs["ylim"]
    ncol = kwargs["ncol"]
    y_ticks_format = kwargs["y_ticks_format"]
    if "timestring" in kwargs:
        save_timestring = kwargs["timestring"]
    else:
        save_timestring = get_now_string()
    # model = kwargs["model"]
    default_color_list = ["green", "red", "blue", "orange", "purple"]
    default_color_list_alpha = ["lime", "pink", "cyan", "gold", "violet"]

    save_folder = "./paper_figure/{}_{}/".format(model_name_short, save_timestring)
    print("saved to {}".format(save_folder))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_loss_nmse_path = "{}/nmse.png".format(save_folder)

    plt.figure(figsize=(8, 6))
    mask = np.asarray([mask_gap * item for item in range(epoch_max // mask_gap)])

    x = None
    for i, one_model_group in enumerate(timestring_dict.keys()):
        timestring_list = timestring_dict[one_model_group]
        info_path_format = info_path_format_dict[one_model_group]
        print(one_model_group)
        loss_collect = []
        for one_timestring in timestring_list:
            info_path = info_path_format.format(one_timestring)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            # print(info["seed"], sum(info["real_loss_nmse"][-5000:])/5000)
            loss_collect.append(np.expand_dims(smooth_conv(info["real_loss_nmse"], kernel_size=kernel_size), axis=0))
        ys = np.concatenate(loss_collect)
        y_mean = np.mean(ys, axis=0)[mask]
        # y_std = smooth_conv(np.std(ys, axis=0), kernel_size=2000)[:-1000][mask]

        y_max = np.max(ys, axis=0)[mask]
        y_min = np.min(ys, axis=0)[mask]
        # y_max = y_mean + y_std
        # y_min = y_mean - y_std
        y_mean = np.log10(y_mean)
        y_max = np.log10(y_max)
        y_min = np.log10(y_min)
        # print(y_mean[:10])
        # print(y_max[:10])
        # print(y_min[:10])
        # print(y_mean.shape)
        # print(y_max.shape)
        # print(y_min.shape)
        x = np.asarray(range(len(y_mean)))
        y_mean = y_mean[kernel_size // 2: -kernel_size // 2]
        y_max = y_max[kernel_size // 2: -kernel_size // 2]
        y_min = y_min[kernel_size // 2: -kernel_size // 2]
        x = x[kernel_size // 2: -kernel_size // 2]
        # print(ys.shape)
        plt.plot(x * mask_gap, y_mean, c=default_color_list[i], linewidth=1, label=one_model_group)
        plt.fill_between(x * mask_gap, y_min, y_max, facecolor=default_color_list_alpha[i], alpha=0.2, linewidth=0)  # edgecolor="black",
    if ylim is not None:
        plt.ylim(ylim)
    if y_ticks is not None:
        plt.yticks(y_ticks, [y_ticks_format % item for item in y_ticks])

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=ncol, fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(save_loss_nmse_path, dpi=500)
    # plt.show()
    plt.close()


def one_time_plot_sir():
    model_name_short = "SIR"
    draw_paper_figure_loss(
        timestring_dict={
            "PINN": ["20221229_143735", "20221229_144457", "20221229_145231", "20221229_145959", "20221229_152210", "20221229_152941"],
            "FNN": ["20221229_143214", "20221229_144129", "20221229_145111", "20221229_150059", "20221229_151040", "20221229_152040", "20221229_153022", "20221229_153948", "20221229_154852"],
            "SB-FNN(A)": ["20221229_143657", "20221229_145101", "20221229_150514", "20221229_151911", "20221229_153320", "20221229_154726", "20221229_160123", "20221229_161517", "20221229_162909"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=2000,
        mask_gap=1,
        epoch_max=20000,
        y_ticks=[-6.0 + 1 * item for item in range(4)],
        ylim=[-6.5, -4.5],
        y_ticks_format="$10^{%d}$",
        ncol=3,
    )

def one_time_plot_turing():
    model_name_short = "Turing"
    draw_paper_figure_loss(
        timestring_dict={
            "PINN": ["20221227_205000", "20221228_005108", "20221228_045255", "20221228_085553", "20221228_125128", "20221228_164823", "20221228_204721", "20221229_004407", "20221229_044122", "20221229_084042"],
            "FNN": ["20221228_001505", "20221228_002600", "20221228_003700", "20221228_004759", "20221228_005859", "20221228_010957", "20221228_012054", "20221228_013153", "20221228_014252", "20221228_015351"],
            "SB-FNN(A)": ["20221228_001505", "20221228_003145", "20221228_004826", "20221228_010505", "20221228_012144", "20221228_013823", "20221228_015504", "20221228_021143", "20221228_022823", "20221228_024501"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=500,
        mask_gap=1,
        epoch_max=3000,
        y_ticks=[-1.2 + 0.1 * item for item in range(6)],
        ylim=[-1.25, -0.65],
        y_ticks_format="$10^{%.1f}$",
        ncol=3,
    )


def one_time_plot_rep():
    model_name_short = "REP"
    draw_paper_figure_loss(
        timestring_dict={
            "FNN": ["20221227_182627", "20221227_184011", "20221227_185409", "20221227_190757", "20221227_192158"],
            "SB-FNN(A)": ["20221227_181943", "20221227_183834", "20221227_185802", "20221227_191618", "20221227_193452", "20221227_195419", "20221227_201329", "20221227_203232", "20221227_205125", "20221227_211010"],
            "SB-FNN(P)": ["20221227_181432", "20221227_182817", "20221227_184153", "20221227_185538", "20221227_190923", "20221227_192306", "20221227_193643", "20221227_195015", "20221227_200346", "20221227_201730"],
            "SB-FNN": ["20221227_183737", "20221227_185526", "20221227_191325", "20221227_193155", "20221227_194947", "20221227_200731", "20221227_202538", "20221227_204359", "20221227_210139", "20221227_211936"],
        },
        info_path_format_dict={
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=2000,
        mask_gap=1,
        epoch_max=20000,
        y_ticks=[-4 + 1 * item for item in range(4)],
        ylim=[-4.5, -0.5],
        y_ticks_format="$10^{%d}$",
        ncol=4,
    )

def one_time_plot_cc1():
    model_name_short = "CC1"
    draw_paper_figure_loss(
        timestring_dict={
            "FNN": ["20221229_194306", "20221229_200333", "20221229_200838", "20221229_193822", "20221229_201045", "20221229_194304", "20221229_195307", "20221229_195809"],
            "SB-FNN(A)": ["20221229_210920", "20221229_211653", "20221229_203141", "20221229_213310", "20221229_210828", "20221229_211653", "20221229_213401", "20221229_205915", "20221229_210736"],
            "SB-FNN(P)": ["20221229_194348", "20221229_200515", "20221229_201017", "20221229_200808", "20221229_203313", "20221229_204315", "20221229_204815"],
            "SB-FNN": ["20221229_215821", "20221229_220554", "20221229_212101", "20221229_222128", "20221229_221744", "20221229_222524", "20221229_215615", "20221229_221134", "20221229_221912"],
        },
        info_path_format_dict={
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=200,
        mask_gap=1,
        epoch_max=10000,
        y_ticks=None,#[-4 + 1 * item for item in range(4)],
        ylim=None,#[-4.5, -0.5],
        y_ticks_format="$10^{%.1f}$",
        ncol=4,
    )


if __name__ == "__main__":
    one_time_plot_cc1()
    # one_time_plot_rep()
    # one_time_plot_turing()
    # one_time_plot_sir()
    # a = np.asarray([[1.0,2,3,4,7], [2,3,4,5,6], [3,4,5,6,7], [3,4,5,6,7], [3,4,5,6,5], [4,5,6,7,8], [5,6,7,8,9]])
    # print(MyNumpy.max(a, 1))
    # a = np.asarray([1, 2, 3])
    # b = np.asarray([11, 22, 33])
    # a = np.expand_dims(a, axis=0)
    # b = np.expand_dims(b, axis=0)
    # collect = np.concatenate([a, b])
    # collect_avg = np.mean(collect, axis=0)
    # print(collect_avg)
    # with open("saves/train/PP_PINN_Lambda_20221230_084139/PP_PINN_Lambda_20221230_084139_info.npy", "rb") as f:
    #     info = pickle.load(f)
    # print(info["seed"])
    # clear_reformat(str1)
    # clear_reformat(str2)
    # clear_reformat(str3)
    # clear_reformat(str4)
    pass
