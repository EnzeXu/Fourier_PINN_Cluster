import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils import smooth_conv, MultiSubplotDraw


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

    plt.figure(figsize=(12, 7))
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

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=False, ncol=ncol, fontsize=20)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("N-MSE Loss", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(save_loss_nmse_path, dpi=500)
    # plt.show()
    plt.close()

def draw_paper_figure_best(**kwargs):
    assert_keyword_list = ["timestring_dict", "info_path_format_dict", "model_name_short", "config", "loss_average_length"]
    assert all(item in kwargs for item in assert_keyword_list)
    timestring_dict = kwargs["timestring_dict"]
    info_path_format_dict = kwargs["info_path_format_dict"]
    model_name_short = kwargs["model_name_short"]
    config = kwargs["config"]
    loss_average_length = kwargs["loss_average_length"]
    fontsize = kwargs["fontsize"]
    if "timestring" in kwargs:
        save_timestring = kwargs["timestring"]
    else:
        save_timestring = get_now_string()

    if "show_legend" in kwargs:
        show_legend = kwargs["show_legend"]
    else:
        show_legend = (["{} (predicted)".format(item) for item in config.curve_names] + ["{} (truth)".format(item) for item in config.curve_names])

    default_best_color_list = ["red", "blue", "green", "lime", "orange", "grey", "lightcoral", "brown", "chocolate", "peachpuff", "dodgerblue", "crimson", "pink", "cornflowerblue", "indigo", "navy", "teal", "seagreen", "orchid", "tan", "plum", "purple", "ivory", "oldlace", "silver", "tomato", "peru", "aliceblue"]

    save_folder = "./paper_figure/{}_{}/".format(model_name_short, save_timestring)
    print("saved to {}".format(save_folder))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, one_model_group in enumerate(timestring_dict.keys()):
        timestring_list = timestring_dict[one_model_group]
        info_path_format = info_path_format_dict[one_model_group]
        # print(one_model_group)
        best_info = None
        best_nmse_loss = 1e10
        best_attempt_seed = None
        best_save_path = "{}/{}_best.png".format(save_folder, one_model_group)
        for one_timestring in timestring_list:
            info_path = info_path_format.format(one_timestring)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            tmp_loss_nmse = sum(info["real_loss_nmse"][-loss_average_length:]) / loss_average_length
            if tmp_loss_nmse < best_nmse_loss:
                # print(info["seed"], sum(info["real_loss_nmse"][-5000:])/5000)
                best_info = info
                best_nmse_loss = tmp_loss_nmse
                best_attempt_seed = info["seed"]
        print("best_attempt_seed =", best_attempt_seed)
        y_predict = np.swapaxes(best_info["y_predict"], 0, 1)
        y_truth = np.swapaxes(best_info["y_truth"], 0, 1)
        y = np.concatenate([y_predict, y_truth], 0)
        x = config.t
        print(y.shape, y_predict.shape, y_truth.shape, x.shape)
        # draw_two_dimension(
        #     y_lists=y,
        #     x_list=x,
        #     color_list=default_best_color_list[:2 * config.prob_dim],
        #     line_style_list=["solid"] * config.prob_dim + ["dashed"] * config.prob_dim,
        #     legend_list=[]
        # )
        plt.figure(figsize=(10, 6))
        for j in range(len(y)):
            plt.plot(x, y[j], linewidth=2, c=default_best_color_list[:2 * config.prob_dim][j],
                     linestyle=(["solid"] * config.prob_dim + ["dashed"] * config.prob_dim)[j],
                     label=show_legend[j],
                     )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig(best_save_path, dpi=500)
        # plt.show()
        plt.close()

def draw_paper_figure_best_turing(**kwargs):
    assert_keyword_list = ["timestring_dict", "info_path_format_dict", "model_name_short", "config", "loss_average_length"]
    assert all(item in kwargs for item in assert_keyword_list)
    timestring_dict = kwargs["timestring_dict"]
    info_path_format_dict = kwargs["info_path_format_dict"]
    model_name_short = kwargs["model_name_short"]
    config = kwargs["config"]
    loss_average_length = kwargs["loss_average_length"]
    fontsize = kwargs["fontsize"]
    if "timestring" in kwargs:
        save_timestring = kwargs["timestring"]
    else:
        save_timestring = get_now_string()

    default_best_color_list = ["red", "blue", "green", "lime", "orange", "grey", "lightcoral", "brown", "chocolate", "peachpuff", "dodgerblue", "crimson", "pink", "cornflowerblue", "indigo", "navy", "teal", "seagreen", "orchid", "tan", "plum", "purple", "ivory", "oldlace", "silver", "tomato", "peru", "aliceblue"]

    save_folder = "./paper_figure/{}_{}/".format(model_name_short, save_timestring)
    print("saved to {}".format(save_folder))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, one_model_group in enumerate(timestring_dict.keys()):
        timestring_list = timestring_dict[one_model_group]
        info_path_format = info_path_format_dict[one_model_group]
        # print(one_model_group)
        best_info = None
        best_nmse_loss = 1e10
        best_attempt_seed = None
        best_save_path = "{}/{}_best.png".format(save_folder, one_model_group)
        for one_timestring in timestring_list:
            info_path = info_path_format.format(one_timestring)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            tmp_loss_nmse = sum(info["real_loss_nmse"][-loss_average_length:]) / loss_average_length
            if tmp_loss_nmse < best_nmse_loss:
                # print(info["seed"], sum(info["real_loss_nmse"][-5000:])/5000)
                best_info = info
                best_nmse_loss = tmp_loss_nmse
                best_attempt_seed = info["seed"]
        print("best_attempt_seed =", best_attempt_seed)
        y_predict = best_info["y_predict"][-1]
        y_truth = best_info["y_truth"][-1]
        # y = np.concatenate([y_predict, y_truth], 0)
        # x = config.t
        print(y_predict.shape, y_truth.shape)
        u_last = y_predict[:, :, 0]
        v_last = y_predict[:, :, 1]
        u_last_true = y_truth[:, :, 0]
        v_last_true = y_truth[:, :, 1]
        m = MultiSubplotDraw(row=2, col=2, fig_size=(8, 8), tight_layout_flag=True, show_flag=False, save_flag=True, save_path=best_save_path, save_dpi=500)
        m.add_subplot_turing(
            matrix=u_last,
            v_max=u_last.max(),  # u_last_true.max(),
            v_min=u_last.min(),  # u_last_true.min()
            fig_title_size=20,
            number_label_size=20,
            colorbar=False,
            fig_title="U (predicted)",
            x_ticks_set_flag=True,
            y_ticks_set_flag=True,
            x_ticks=range(0, 30, 5),
            y_ticks=range(0, 30, 5),
            # fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
        )
        m.add_subplot_turing(
            matrix=v_last,
            v_max=v_last.max(),  # v_last_true.max()
            v_min=v_last.min(),  # v_last_true.min()
            fig_title_size=20,
            number_label_size=20,
            colorbar=False,
            fig_title="V (predicted)",
            x_ticks_set_flag=True,
            y_ticks_set_flag=True,
            x_ticks=range(0, 30, 5),
            y_ticks=range(0, 30, 5),
            # fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
        )
        m.add_subplot_turing(
            matrix=u_last_true,
            v_max=u_last_true.max(),
            v_min=u_last_true.min(),
            fig_title_size=20,
            number_label_size=20,
            colorbar=False,
            fig_title="U (truth)",
            x_ticks_set_flag=True,
            y_ticks_set_flag=True,
            x_ticks=range(0, 30, 5),
            y_ticks=range(0, 30, 5),
            # fig_title="{}_{}_U_true".format(self.config.model_name, self.time_string)
        )
        m.add_subplot_turing(
            matrix=v_last_true,
            v_max=v_last_true.max(),
            v_min=v_last_true.min(),
            fig_title_size=20,
            number_label_size=20,
            colorbar=False,
            fig_title="V (truth)",
            x_ticks_set_flag=True,
            y_ticks_set_flag=True,
            x_ticks=range(0, 30, 5),
            y_ticks=range(0, 30, 5),
            # fig_title="{}_{}_V_true".format(self.config.model_name, self.time_string)
        )
        m.draw()

def one_time_plot_sir_best():
    model_name_short = "SIR"
    from model_SIR_Lambda import Config
    draw_paper_figure_best(
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
        config=Config(),
        loss_average_length=5000,
        fontsize=20,
    )

def one_time_plot_rep_best():
    model_name_short = "REP"
    from model_REP_Lambda import Config
    draw_paper_figure_best(
        timestring_dict={
            "PINN": ["20221227_170905", "20221227_172359", "20221227_173851", "20221227_175341", "20221227_180829", "20221227_182326", "20221227_183830", "20221227_185324", "20221227_190825", "20221227_192326"],
            "FNN": ["20221227_182627", "20221227_184011", "20221227_185409", "20221227_190757", "20221227_192158"],
            "SB-FNN(A)": ["20221227_181943", "20221227_183834", "20221227_185802", "20221227_191618", "20221227_193452", "20221227_195419", "20221227_201329", "20221227_203232", "20221227_205125", "20221227_211010"],
            "SB-FNN(P)": ["20221227_181432", "20221227_182817", "20221227_184153", "20221227_185538", "20221227_190923", "20221227_192306", "20221227_193643", "20221227_195015", "20221227_200346", "20221227_201730"],
            "SB-FNN": ["20221227_183737", "20221227_185526", "20221227_191325", "20221227_193155", "20221227_194947", "20221227_200731", "20221227_202538", "20221227_204359", "20221227_210139", "20221227_211936"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        config=Config(),
        loss_average_length=5000,
        fontsize=20,
    )

def one_time_plot_cc1_best():
    model_name_short = "CC1"
    from model_CC1_Lambda import Config
    draw_paper_figure_best(
        timestring_dict={
            "PINN": ["20221229_214847", "20221229_215257", "20221229_215701", "20221229_220110", "20221229_220510", "20221229_220906", "20221229_221300", "20221229_221707", "20221229_222122", "20221229_222524", "20221229_215045", "20221229_215443", "20221229_215848", "20221229_220259", "20221229_220701", "20221229_221103", "20221229_221519", "20221229_221923", "20221229_222320", "20221229_222718", "20221229_222829", "20221229_223213", "20221229_223556", "20221229_223940", "20221229_224326", "20221229_224710", "20221229_225054", "20221229_225439", "20221229_225823", "20221229_230207", "20221229_222900", "20221229_223247", "20221229_223633", "20221229_224018", "20221229_224403", "20221229_224746", "20221229_225129", "20221229_225514", "20221229_225858", "20221229_230242"],
            "FNN": ["20221229_194306", "20221229_200333", "20221229_200838", "20221229_193822", "20221229_201045", "20221229_194304", "20221229_195307", "20221229_195809"],
            "SB-FNN(A)": ["20221229_210920", "20221229_211653", "20221229_203141", "20221229_213310", "20221229_210828", "20221229_211653", "20221229_213401", "20221229_205915", "20221229_210736"],
            "SB-FNN(P)": ["20221229_194348", "20221229_200515", "20221229_201017", "20221229_200808", "20221229_203313", "20221229_204315", "20221229_204815"],
            "SB-FNN": ["20221229_215821", "20221229_220554", "20221229_212101", "20221229_222128", "20221229_221744", "20221229_222524", "20221229_215615", "20221229_221134", "20221229_221912"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        config=Config(),
        loss_average_length=1000,
        fontsize=20,
    )

def one_time_plot_turing_best():
    model_name_short = "Turing"
    from model_CC1_Lambda import Config
    draw_paper_figure_best_turing(
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
        config=Config(),
        loss_average_length=1000,
        fontsize=20,
    )

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
    from model_REP_Lambda import Config
    model_name_short = "REP"
    time_string = get_now_string()
    draw_paper_figure_loss(
        timestring_dict={
            "PINN": ["20230113_072355_474015", "20230113_075729_957285", "20230113_083048_636737", "20230113_090445_351975", "20230113_093935_520438", "20230113_101402_523720", "20230113_104742_611608", "20230113_112205_274969", "20230113_115755_624347", "20230113_123231_431485"],
            "FNN": ["20230113_075739_832183", "20230113_083929_235082", "20230113_092123_837192", "20230113_100401_481913", "20230113_104713_061893", "20230113_112931_198531", "20230113_121254_870794", "20230113_125559_423198", "20230113_134041_313176", "20230113_142505_384061"],
            "SB-FNN(A)": ["20230113_104739_849772", "20230113_114307_077581", "20230113_123924_947628", "20230113_133439_053026", "20230113_143009_620944", "20230113_152528_950426", "20230113_162054_882916", "20230113_171632_140936", "20230113_181200_480415", "20230113_190758_193187"],
            "SB-FNN(P)": ["20230113_085528_076951", "20230113_094201_864914", "20230113_102937_708564", "20230113_111531_049110", "20230113_120023_126623", "20230113_124608_611381", "20230113_133212_229659", "20230113_141806_950491", "20230113_150305_232567", "20230113_154838_167886"],
            "SB-FNN": ["20230113_130616_626987", "20230113_140217_611121", "20230113_145746_666743", "20230113_155742_700314", "20230113_165401_732762", "20230113_175040_795192", "20230113_184641_305244", "20230113_194322_514807", "20230113_203951_160971", "20230113_213559_235176"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=2000,
        mask_gap=1,
        epoch_max=50000,
        y_ticks=[-5 + 1 * item for item in range(9)],
        ylim=[-5.5, 3.1],
        y_ticks_format="$10^{%d}$",
        ncol=5,
        timestring=time_string,
    )
    draw_paper_figure_best(
        timestring_dict={
            "PINN": ["20230113_072355_474015", "20230113_075729_957285", "20230113_083048_636737", "20230113_090445_351975", "20230113_093935_520438", "20230113_101402_523720", "20230113_104742_611608", "20230113_112205_274969", "20230113_115755_624347", "20230113_123231_431485"],
            "FNN": ["20230113_075739_832183", "20230113_083929_235082", "20230113_092123_837192", "20230113_100401_481913", "20230113_104713_061893", "20230113_112931_198531", "20230113_121254_870794", "20230113_125559_423198", "20230113_134041_313176", "20230113_142505_384061"],
            "SB-FNN(A)": ["20230113_104739_849772", "20230113_114307_077581", "20230113_123924_947628", "20230113_133439_053026", "20230113_143009_620944", "20230113_152528_950426", "20230113_162054_882916", "20230113_171632_140936", "20230113_181200_480415", "20230113_190758_193187"],
            "SB-FNN(P)": ["20230113_085528_076951", "20230113_094201_864914", "20230113_102937_708564", "20230113_111531_049110", "20230113_120023_126623", "20230113_124608_611381", "20230113_133212_229659", "20230113_141806_950491", "20230113_150305_232567", "20230113_154838_167886"],
            "SB-FNN": ["20230113_130616_626987", "20230113_140217_611121", "20230113_145746_666743", "20230113_155742_700314", "20230113_165401_732762", "20230113_175040_795192", "20230113_184641_305244", "20230113_194322_514807", "20230113_203951_160971", "20230113_213559_235176"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        config=Config(),
        loss_average_length=5000,
        fontsize=20,
        timestring=time_string,
        show_legend=["$\widehat{{{}}}$".format(item) for item in ["M_{lacI}", "M_{tetR}", "M_{cI}", "P_{cI}", "P_{lacI}", "P_{tetR}"]] + ["${}$".format(item) for item in ["M_{lacI}", "M_{tetR}", "M_{cI}", "P_{cI}", "P_{lacI}", "P_{tetR}"]]
    )

def one_time_plot_cc1():
    from model_CC1_Lambda import Config
    model_name_short = "CC1"
    time_string = get_now_string()
    draw_paper_figure_loss(
        timestring_dict={
            "PINN": ["20230114_104738_082733", "20230114_110633_738392", "20230114_112534_011307", "20230114_114419_776550", "20230114_120321_408292", "20230114_122211_757770", "20230114_124112_930914", "20230114_125956_718534", "20230114_131928_319098", "20230114_133828_281110", "20230114_105828_462987", "20230114_111836_784358", "20230114_113847_415229", "20230114_115906_425832", "20230114_121916_339738", "20230114_123910_001294", "20230114_130014_384850", "20230114_132027_156401", "20230114_134018_860625", "20230114_140047_470748"],
            "FNN": ["20230113_150114_122814", "20230113_170207_774184", "20230113_173218_621612", "20230113_180213_136149", "20230113_153824_742465", "20230113_174035_034880", "20230113_181000_331963", "20230113_184102_918462", "20230113_191619_207010", "20230113_194800_328890", "20230113_190234_884004", "20230113_200256_223610", "20230113_210323_501140", "20230113_213341_501826", "20230113_233409_823585"],
            "SB-FNN(A)": ["20230114_042652_455355", "20230114_051054_533232", "20230114_014100_517379", "20230114_043138_706828", "20230114_072151_686295", "20230114_045400_054312", "20230114_053845_970724", "20230114_070401_975744", "20230114_035946_702151", "20230114_044917_964045", "20230114_062205_601996", "20230114_070728_045072", "20230114_083957_451761", "20230114_101241_352682"],
            "SB-FNN(P)": ["20230113_200410_615541", "20230113_210247_686231", "20230113_230054_282851", "20230113_233021_933410", "20230113_235948_661313", "20230113_204321_910480", "20230113_224559_812952", "20230113_231609_139457", "20230113_234559_811241", "20230114_003354_004737", "20230114_010603_573678", "20230113_223126_956416", "20230113_233159_014195", "20230114_003102_913432", "20230114_010019_176617", "20230114_030006_786992"],
            "SB-FNN": ["20230114_075356_293959", "20230114_083750_174536", "20230114_080525_254861", "20230114_110112_844998", "20230114_131356_291274", "20230114_135752_431809", "20230114_084711_173631", "20230114_113744_155337", "20230114_122001_852017", "20230114_134448_521618", "20230114_091341_089367", "20230114_095622_967548", "20230114_112156_646795", "20230114_120428_577061", "20230114_133048_012766", "20230114_145713_453669"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=2000,
        mask_gap=1,
        epoch_max=50000,
        y_ticks=[-6 + 1 * item for item in range(9)],
        ylim=[-6.5, 2.5],
        y_ticks_format="$10^{%d}$",
        ncol=5,
        timestring=time_string,
    )
    draw_paper_figure_best(
        timestring_dict={
            "PINN": ["20230114_104738_082733", "20230114_110633_738392", "20230114_112534_011307", "20230114_114419_776550", "20230114_120321_408292", "20230114_122211_757770", "20230114_124112_930914", "20230114_125956_718534", "20230114_131928_319098", "20230114_133828_281110", "20230114_105828_462987", "20230114_111836_784358", "20230114_113847_415229", "20230114_115906_425832", "20230114_121916_339738", "20230114_123910_001294", "20230114_130014_384850", "20230114_132027_156401", "20230114_134018_860625", "20230114_140047_470748"],
            "FNN": ["20230113_150114_122814", "20230113_170207_774184", "20230113_173218_621612", "20230113_180213_136149", "20230113_153824_742465", "20230113_174035_034880", "20230113_181000_331963", "20230113_184102_918462", "20230113_191619_207010", "20230113_194800_328890", "20230113_190234_884004", "20230113_200256_223610", "20230113_210323_501140", "20230113_213341_501826", "20230113_233409_823585"],
            "SB-FNN(A)": ["20230114_042652_455355", "20230114_051054_533232", "20230114_014100_517379", "20230114_043138_706828", "20230114_072151_686295", "20230114_045400_054312", "20230114_053845_970724", "20230114_070401_975744", "20230114_035946_702151", "20230114_044917_964045", "20230114_062205_601996", "20230114_070728_045072", "20230114_083957_451761", "20230114_101241_352682"],
            "SB-FNN(P)": ["20230113_200410_615541", "20230113_210247_686231", "20230113_230054_282851", "20230113_233021_933410", "20230113_235948_661313", "20230113_204321_910480", "20230113_224559_812952", "20230113_231609_139457", "20230113_234559_811241", "20230114_003354_004737", "20230114_010603_573678", "20230113_223126_956416", "20230113_233159_014195", "20230114_003102_913432", "20230114_010019_176617", "20230114_030006_786992"],
            "SB-FNN": ["20230114_075356_293959", "20230114_083750_174536", "20230114_080525_254861", "20230114_110112_844998", "20230114_131356_291274", "20230114_135752_431809", "20230114_084711_173631", "20230114_113744_155337", "20230114_122001_852017", "20230114_134448_521618", "20230114_091341_089367", "20230114_095622_967548", "20230114_112156_646795", "20230114_120428_577061", "20230114_133048_012766", "20230114_145713_453669"],
        },
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
            "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
            "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
                model_name_short),
            "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
                model_name_short),
            "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
                model_name_short),
        },
        model_name_short=model_name_short,
        config=Config(),
        loss_average_length=1000,
        fontsize=20,
        timestring=time_string,
        show_legend=["$\widehat{{{}}}$".format(item) for item in ["MPF", "Kin_{p}", "APC_{p}"]] + ["${}$".format(item) for item in ["MPF", "Kin_{p}", "APC_{p}"]],
    )

str0 = """
 20230114_104738_082733
 20230114_110633_738392
 20230114_112534_011307
 20230114_114419_776550
 20230114_120321_408292
 20230114_122211_757770
 20230114_124112_930914
 20230114_125956_718534
 20230114_131928_319098
 20230114_133828_281110
 20230114_105828_462987
 20230114_111836_784358
 20230114_113847_415229
 20230114_115906_425832
 20230114_121916_339738
 20230114_123910_001294
 20230114_130014_384850
 20230114_132027_156401
 20230114_134018_860625
 20230114_140047_470748
"""

str1 = """
 20230113_150114_122814
 20230113_170207_774184
 20230113_173218_621612
 20230113_180213_136149
 20230113_153824_742465
 20230113_174035_034880
 20230113_181000_331963
 20230113_184102_918462
 20230113_191619_207010
 20230113_194800_328890
 20230113_190234_884004
 20230113_200256_223610
 20230113_210323_501140
 20230113_213341_501826
 20230113_233409_823585
"""

str2 = """
 20230114_042652_455355
 20230114_051054_533232
 20230114_014100_517379
 20230114_043138_706828
 20230114_072151_686295
 20230114_045400_054312
 20230114_053845_970724
 20230114_070401_975744
 20230114_035946_702151
 20230114_044917_964045
 20230114_062205_601996
 20230114_070728_045072
 20230114_083957_451761
 20230114_101241_352682
"""

str3 = """
 20230113_200410_615541
 20230113_210247_686231
 20230113_230054_282851
 20230113_233021_933410
 20230113_235948_661313
 20230113_204321_910480
 20230113_224559_812952
 20230113_231609_139457
 20230113_234559_811241
 20230114_003354_004737
 20230114_010603_573678
 20230113_223126_956416
 20230113_233159_014195
 20230114_003102_913432
 20230114_010019_176617
 20230114_030006_786992
"""

str4 = """
 20230114_075356_293959
 20230114_083750_174536
 20230114_080525_254861
 20230114_110112_844998
 20230114_131356_291274
 20230114_135752_431809
 20230114_084711_173631
 20230114_113744_155337
 20230114_122001_852017
 20230114_134448_521618
 20230114_091341_089367
 20230114_095622_967548
 20230114_112156_646795
 20230114_120428_577061
 20230114_133048_012766
 20230114_145713_453669
"""



if __name__ == "__main__":
    # one_time_plot_turing_best()
    # one_time_plot_sir_best()
    # one_time_plot_rep_best()
    # one_time_plot_cc1_best()
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
    # clear_reformat(str0)
    # clear_reformat(str1)
    # clear_reformat(str2)
    # clear_reformat(str3)
    # clear_reformat(str4)
    pass
