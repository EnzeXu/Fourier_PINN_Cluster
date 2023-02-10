import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils import smooth_conv, MultiSubplotDraw, ColorCandidate


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

def draw_turing(u, v, save_flag=False, save_path=None):
    fig = plt.figure(figsize=(24, 5))

    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(u, cmap=plt.cm.jet, vmin=u[:,-1:].min(), vmax=u[:,-1:].max(), aspect='auto')
    ax1.set_title("u")
    cb1 = plt.colorbar(im1, shrink=1)

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(v, cmap=plt.cm.jet, vmin=v[:,-1:].min(), vmax=v[:,-1:].max(), aspect='auto')
    ax2.set_title("v")
    cb2 = plt.colorbar(im2, shrink=1)
    plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.clf()


def one_time_draw_turing1d(timestring):
    model_name_short = "Turing1D"
    info_path = "./saves/train/{0}_Fourier_Omega_{1}/{0}_Fourier_Omega_{1}_info.npy".format(model_name_short, timestring)
    with open(info_path, "rb") as f:
        info = pickle.load(f)
    print(info["y_predict"].shape)
    print(info["y_truth"].shape)
    u_predict = np.swapaxes(info["y_predict"][:, :, :, 0].reshape([info["y_predict"].shape[0], info["y_predict"].shape[1]]), 0, 1)
    v_predict = np.swapaxes(info["y_predict"][:, :, :, 1].reshape([info["y_predict"].shape[0], info["y_predict"].shape[1]]), 0, 1)
    u_truth = np.swapaxes(info["y_truth"][:, :, :, 0].reshape([info["y_truth"].shape[0], info["y_truth"].shape[1]]), 0, 1)
    v_truth = np.swapaxes(info["y_truth"][:, :, :, 1].reshape([info["y_truth"].shape[0], info["y_truth"].shape[1]]), 0, 1)
    draw_turing(u_predict, v_predict)
    draw_turing(u_truth, v_truth)

def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def clear_reformat(string):
    lines = string.split()
    lines = ["\"{}\"".format(item) for item in lines if len(item) > 2]
    res_string = "[{}],".format(", ".join(lines))
    print(res_string)

def clear_reformat_dictionary(dic):
    for one_key in dic:
        lines = dic[one_key].split()
        lines = ["\"{}\"".format(item) for item in lines if len(item) > 2]
        res_string = "[{}],".format(", ".join(lines))
        print("\"{}\":".format(one_key), res_string)


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
    skip_log_flag = False if "skip_log_flag" not in kwargs else kwargs["skip_log_flag"]
    start_index = 0 if "skip_log_flag" not in kwargs else kwargs["start_index"]
    if "legend_fontsize" in kwargs:
        legend_fontsize = kwargs["legend_fontsize"]
    else:
        legend_fontsize = 30
    y_ticks_format = kwargs["y_ticks_format"]
    if "timestring" in kwargs:
        save_timestring = kwargs["timestring"]
    else:
        save_timestring = get_now_string()
    # model = kwargs["model"]
    color_n = len(timestring_dict.keys())
    default_colors = ColorCandidate().get_color_list(color_n, 0.5)
    default_color_list = default_colors[:color_n]
    default_color_list_alpha = default_colors[-color_n:]

    save_folder = "./paper_figure/{}_{}/".format(model_name_short if not kwargs["all_flag"] else "All", save_timestring)
    print("{}: saved to {}".format(model_name_short, save_folder))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_loss_nmse_path = "{}/nmse_loss_{}.png".format(save_folder, model_name_short.lower())

    plt.figure(figsize=(12, 6))
    mask = np.asarray([mask_gap * item for item in range(epoch_max // mask_gap)])

    x = None
    for i, one_model_group in enumerate(timestring_dict.keys()):
        timestring_list = timestring_dict[one_model_group]
        if one_model_group not in info_path_format_dict:
            info_path_format = info_path_format_dict["default"]
        else:
            info_path_format = info_path_format_dict[one_model_group]
        print(one_model_group)
        loss_collect = []
        for one_timestring in timestring_list:
            if len(one_timestring) == 0:
                continue
            info_path = info_path_format.format(one_timestring)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            # print(info["seed"], sum(info["real_loss_nmse"][-5000:])/5000)
            loss_collect.append(np.expand_dims(smooth_conv(info["real_loss_nmse"], kernel_size=kernel_size), axis=0))
            # print((np.expand_dims(smooth_conv(info["real_loss_nmse"], kernel_size=kernel_size), axis=0)).shape)
        ys = np.concatenate(loss_collect)
        y_mean = np.mean(ys, axis=0)[mask]
        # y_std = smooth_conv(np.std(ys, axis=0), kernel_size=2000)[:-1000][mask]

        y_max = np.max(ys, axis=0)[mask]
        y_min = np.min(ys, axis=0)[mask]
        # y_max = y_mean + y_std
        # y_min = y_mean - y_std
        if model_name_short != "Turing" and not skip_log_flag:
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
        plt.plot((x * mask_gap)[start_index:], y_mean[start_index:], c=default_color_list[i], linewidth=3, label=one_model_group, alpha=0.5)
        plt.fill_between((x * mask_gap)[start_index:], y_min[start_index:], y_max[start_index:], facecolor=default_color_list_alpha[i], alpha=0.2, linewidth=0)  # edgecolor="black",
    if ylim is not None:
        plt.ylim(ylim)
    if y_ticks is not None:
        plt.yticks(y_ticks, [y_ticks_format % item for item in y_ticks])
    if len(timestring_dict.keys()) >= 99:#4:
        bbox_to_anchor = (0.5, 1.40)
        ncol = 3
        legend_fontsize = 15
    else:
        bbox_to_anchor = (0.5, 1.30)
    plt.legend(loc="upper center", bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=False, ncol=ncol, fontsize=legend_fontsize)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.xlabel("Epoch", fontsize=30)
    plt.ylabel("N-MSE Loss", fontsize=30)
    plt.tick_params(labelsize=30)
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

    # default_best_color_list = ["red", "blue", "green", "lime", "orange", "grey", "lightcoral", "brown", "chocolate", "peachpuff", "dodgerblue", "crimson", "pink", "cornflowerblue", "indigo", "navy", "teal", "seagreen", "orchid", "tan", "plum", "purple", "ivory", "oldlace", "silver", "tomato", "peru", "aliceblue"]
    default_best_color_list = ColorCandidate().get_color_list(config.prob_dim, 0.5)
    save_folder = "./paper_figure/{}_{}/".format(model_name_short if not kwargs["all_flag"] else "All", save_timestring)
    print("{}: saved to {}".format(model_name_short, save_folder))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, one_model_group in enumerate(timestring_dict.keys()):
        timestring_list = timestring_dict[one_model_group]
        if one_model_group not in info_path_format_dict:
            info_path_format = info_path_format_dict["default"]
        else:
            info_path_format = info_path_format_dict[one_model_group]
        # print(one_model_group)
        best_info = None
        best_nmse_loss = 1e10
        best_attempt_seed = None
        best_save_path = "{}/best_{}_{}.png".format(save_folder, one_model_group, model_name_short.lower())
        for one_timestring in timestring_list:
            if len(one_timestring) == 0:
                continue
            info_path = info_path_format.format(one_timestring)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            tmp_loss_nmse = sum(info["real_loss_nmse"][-loss_average_length:]) / loss_average_length
            if tmp_loss_nmse < best_nmse_loss:
                # print(info["seed"], sum(info["real_loss_nmse"][-5000:])/5000)
                best_info = info
                best_nmse_loss = tmp_loss_nmse
                best_attempt_seed = info["seed"]
        print("{}: best_attempt_seed = {}".format(one_model_group, best_attempt_seed))
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
        plt.figure(figsize=(12, 6))
        for j in range(len(y)):
            plt.plot(x, y[j], linewidth=2, c=default_best_color_list[:2 * config.prob_dim][j],
                     linestyle=(["solid"] * config.prob_dim + ["dashed"] * config.prob_dim)[j],
                     label=show_legend[j],
                     )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.xlabel("Time", fontsize=30)
        # plt.xlabel("Value", fontsize=20)
        plt.tick_params(labelsize=30)
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

    save_folder = "./paper_figure/{}_{}/".format(model_name_short if not kwargs["all_flag"] else "All", save_timestring)
    print("{}: saved to {}".format(model_name_short, save_folder))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, one_model_group in enumerate(timestring_dict.keys()):
        timestring_list = timestring_dict[one_model_group]
        if one_model_group not in info_path_format_dict:
            info_path_format = info_path_format_dict["default"]
        else:
            info_path_format = info_path_format_dict[one_model_group]
        # print(one_model_group)
        best_info = None
        best_nmse_loss = 1e10
        best_attempt_seed = None
        best_save_path = "{}/best_{}_{}.png".format(save_folder, one_model_group, model_name_short.lower())
        for one_timestring in timestring_list:
            if len(one_timestring) == 0:
                continue
            info_path = info_path_format.format(one_timestring)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            tmp_loss_nmse = sum(info["real_loss_nmse"][-loss_average_length:]) / loss_average_length
            if tmp_loss_nmse < best_nmse_loss:
                # print(info["seed"], sum(info["real_loss_nmse"][-5000:])/5000)
                best_info = info
                best_nmse_loss = tmp_loss_nmse
                best_attempt_seed = info["seed"]
        print("{}: best_attempt_seed = {}".format(one_model_group, best_attempt_seed))
        y_predict = best_info["y_predict"][-1]
        y_truth = best_info["y_truth"][-1]
        # y = np.concatenate([y_predict, y_truth], 0)
        # x = config.t
        # print(y_predict.shape, y_truth.shape)
        u_last = y_predict[:, :, 0]
        v_last = y_predict[:, :, 1]
        # u_last_true = y_truth[:, :, 0]
        # v_last_true = y_truth[:, :, 1]
        m = MultiSubplotDraw(row=1, col=2, fig_size=(12, 6), tight_layout_flag=True, show_flag=False, save_flag=True, save_path=best_save_path, save_dpi=500)
        m.add_subplot_turing(
            matrix=u_last,
            v_max=u_last.max(),  # u_last_true.max(),
            v_min=u_last.min(),  # u_last_true.min()
            fig_title_size=30,
            number_label_size=30,
            colorbar=False,
            fig_title="$\hat{U}$",
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
            fig_title_size=30,
            number_label_size=30,
            colorbar=False,
            fig_title="$\hat{V}$",
            x_ticks_set_flag=True,
            y_ticks_set_flag=True,
            x_ticks=range(0, 30, 5),
            y_ticks=range(0, 30, 5),
            # fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
        )
        # m.add_subplot_turing(
        #     matrix=u_last_true,
        #     v_max=u_last_true.max(),
        #     v_min=u_last_true.min(),
        #     fig_title_size=20,
        #     number_label_size=20,
        #     colorbar=False,
        #     fig_title="U (truth)",
        #     x_ticks_set_flag=True,
        #     y_ticks_set_flag=True,
        #     x_ticks=range(0, 30, 5),
        #     y_ticks=range(0, 30, 5),
        #     # fig_title="{}_{}_U_true".format(self.config.model_name, self.time_string)
        # )
        # m.add_subplot_turing(
        #     matrix=v_last_true,
        #     v_max=v_last_true.max(),
        #     v_min=v_last_true.min(),
        #     fig_title_size=20,
        #     number_label_size=20,
        #     colorbar=False,
        #     fig_title="V (truth)",
        #     x_ticks_set_flag=True,
        #     y_ticks_set_flag=True,
        #     x_ticks=range(0, 30, 5),
        #     y_ticks=range(0, 30, 5),
        #     # fig_title="{}_{}_V_true".format(self.config.model_name, self.time_string)
        # )
        m.draw()

# def one_time_plot_sir_best():
#     model_name_short = "SIR"
#     from model_SIR_Lambda import Config
#     draw_paper_figure_best(
#         timestring_dict={
#             "PINN": ["20221229_143735", "20221229_144457", "20221229_145231", "20221229_145959", "20221229_152210", "20221229_152941"],
#             "FNN": ["20221229_143214", "20221229_144129", "20221229_145111", "20221229_150059", "20221229_151040", "20221229_152040", "20221229_153022", "20221229_153948", "20221229_154852"],
#             "SB-FNN(A)": ["20221229_143657", "20221229_145101", "20221229_150514", "20221229_151911", "20221229_153320", "20221229_154726", "20221229_160123", "20221229_161517", "20221229_162909"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=5000,
#         fontsize=20,
#     )
#
# def one_time_plot_rep_best():
#     model_name_short = "REP"
#     from model_REP_Lambda import Config
#     draw_paper_figure_best(
#         timestring_dict={
#             "PINN": ["20221227_170905", "20221227_172359", "20221227_173851", "20221227_175341", "20221227_180829", "20221227_182326", "20221227_183830", "20221227_185324", "20221227_190825", "20221227_192326"],
#             "FNN": ["20221227_182627", "20221227_184011", "20221227_185409", "20221227_190757", "20221227_192158"],
#             "SB-FNN(A)": ["20221227_181943", "20221227_183834", "20221227_185802", "20221227_191618", "20221227_193452", "20221227_195419", "20221227_201329", "20221227_203232", "20221227_205125", "20221227_211010"],
#             "SB-FNN(P)": ["20221227_181432", "20221227_182817", "20221227_184153", "20221227_185538", "20221227_190923", "20221227_192306", "20221227_193643", "20221227_195015", "20221227_200346", "20221227_201730"],
#             "SB-FNN": ["20221227_183737", "20221227_185526", "20221227_191325", "20221227_193155", "20221227_194947", "20221227_200731", "20221227_202538", "20221227_204359", "20221227_210139", "20221227_211936"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=5000,
#         fontsize=20,
#     )
#
# def one_time_plot_cc1_best():
#     model_name_short = "CC1"
#     from model_CC1_Lambda import Config
#     draw_paper_figure_best(
#         timestring_dict={
#             "PINN": ["20221229_214847", "20221229_215257", "20221229_215701", "20221229_220110", "20221229_220510", "20221229_220906", "20221229_221300", "20221229_221707", "20221229_222122", "20221229_222524", "20221229_215045", "20221229_215443", "20221229_215848", "20221229_220259", "20221229_220701", "20221229_221103", "20221229_221519", "20221229_221923", "20221229_222320", "20221229_222718", "20221229_222829", "20221229_223213", "20221229_223556", "20221229_223940", "20221229_224326", "20221229_224710", "20221229_225054", "20221229_225439", "20221229_225823", "20221229_230207", "20221229_222900", "20221229_223247", "20221229_223633", "20221229_224018", "20221229_224403", "20221229_224746", "20221229_225129", "20221229_225514", "20221229_225858", "20221229_230242"],
#             "FNN": ["20221229_194306", "20221229_200333", "20221229_200838", "20221229_193822", "20221229_201045", "20221229_194304", "20221229_195307", "20221229_195809"],
#             "SB-FNN(A)": ["20221229_210920", "20221229_211653", "20221229_203141", "20221229_213310", "20221229_210828", "20221229_211653", "20221229_213401", "20221229_205915", "20221229_210736"],
#             "SB-FNN(P)": ["20221229_194348", "20221229_200515", "20221229_201017", "20221229_200808", "20221229_203313", "20221229_204315", "20221229_204815"],
#             "SB-FNN": ["20221229_215821", "20221229_220554", "20221229_212101", "20221229_222128", "20221229_221744", "20221229_222524", "20221229_215615", "20221229_221134", "20221229_221912"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=1000,
#         fontsize=20,
#     )
#
# def one_time_plot_turing_best():
#     model_name_short = "Turing"
#     from model_CC1_Lambda import Config
#     draw_paper_figure_best_turing(
#         timestring_dict={
#             "PINN": ["20221227_205000", "20221228_005108", "20221228_045255", "20221228_085553", "20221228_125128", "20221228_164823", "20221228_204721", "20221229_004407", "20221229_044122", "20221229_084042"],
#             "FNN": ["20221228_001505", "20221228_002600", "20221228_003700", "20221228_004759", "20221228_005859", "20221228_010957", "20221228_012054", "20221228_013153", "20221228_014252", "20221228_015351"],
#             "SB-FNN(A)": ["20221228_001505", "20221228_003145", "20221228_004826", "20221228_010505", "20221228_012144", "20221228_013823", "20221228_015504", "20221228_021143", "20221228_022823", "20221228_024501"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=1000,
#         fontsize=20,
#     )

# def one_time_plot_sir(time_string=None, all_flag=False):
#     from model_SIR_Lambda import Config
#     model_name_short = "SIR"
#     time_string = get_now_string() if not time_string else time_string
#     draw_paper_figure_loss(
#         timestring_dict={
#             "PINN": ["20230113_044709_702458", "20230113_050615_410338", "20230113_052520_287366", "20230113_054353_025943", "20230113_060252_533113", "20230113_062138_780026", "20230113_064057_939906", "20230113_065942_762433", "20230113_071837_172933", "20230113_073717_777907"],
#             "FNN": ["20230113_044709_572369", "20230113_051132_251255", "20230113_053631_684393", "20230113_060128_837339", "20230113_062648_579243", "20230113_065116_747622", "20230113_071609_222132", "20230113_074112_596632", "20230113_080615_761453", "20230113_083124_455906"],
#             "SB-FNN": ["20230113_044709_441633", "20230113_052333_666996", "20230113_055945_709957", "20230113_063546_830813", "20230113_071153_026599", "20230113_074756_116466", "20230113_082338_234823", "20230113_085932_306269", "20230113_093553_230438", "20230113_101153_578230"],
#         },#(A)
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },#(A)
#         model_name_short=model_name_short,
#         kernel_size=5000,
#         mask_gap=1,
#         epoch_max=50000,
#         y_ticks=[-7.0 + 1 * item for item in range(8)],
#         ylim=[-7.5, -0.5],
#         y_ticks_format="$10^{%d}$",
#         ncol=3,
#         timestring=time_string,
#         all_flag=all_flag,
#     )
#     draw_paper_figure_best(
#         timestring_dict={
#             "PINN": ["20230113_044709_702458", "20230113_050615_410338", "20230113_052520_287366",
#                      "20230113_054353_025943", "20230113_060252_533113", "20230113_062138_780026",
#                      "20230113_064057_939906", "20230113_065942_762433", "20230113_071837_172933",
#                      "20230113_073717_777907"],
#             "FNN": ["20230113_044709_572369", "20230113_051132_251255", "20230113_053631_684393",
#                     "20230113_060128_837339", "20230113_062648_579243", "20230113_065116_747622",
#                     "20230113_071609_222132", "20230113_074112_596632", "20230113_080615_761453",
#                     "20230113_083124_455906"],
#             "SB-FNN": ["20230113_044709_441633", "20230113_052333_666996", "20230113_055945_709957",
#                           "20230113_063546_830813", "20230113_071153_026599", "20230113_074756_116466",
#                           "20230113_082338_234823", "20230113_085932_306269", "20230113_093553_230438",
#                           "20230113_101153_578230"],#(A)
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
#                 model_name_short),#(A)
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=5000,
#         fontsize=30,
#         timestring=time_string,
#         show_legend=["$\hat{{{}}}$".format(item) for item in ["S", "I", "R"]] + ["${}$".format(item) for item in ["S", "I", "R"]],
#         all_flag=all_flag,
#     )
#
# def one_time_plot_turing(time_string=None, all_flag=False):
#     from model_Turing_Lambda import Config
#     model_name_short = "Turing"
#     time_string = get_now_string() if not time_string else time_string
#     draw_paper_figure_loss(
#         timestring_dict={
#             "PINN": ["20230113_012543_502301", "20230113_152941_190182", "20230114_050659_463810"],
#             "FNN": ["20230113_012543_388054", "20230113_020130_943114", "20230113_023725_022232", "20230113_031314_069539", "20230113_034902_802106", "20230113_042449_490497", "20230113_050033_762260", "20230113_053627_898635", "20230113_061223_021126", "20230113_064812_984827"],
#             "SB-FNN": ["20230113_044715_851325", "20230113_054236_639695", "20230113_063806_446492", "20230113_073328_129274", "20230113_082852_515273", "20230113_092420_414350", "20230113_101949_559684", "20230113_111508_182022", "20230113_121028_140717", "20230113_130547_991195"],#(A)
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),#(A)
#         },
#         model_name_short=model_name_short,
#         kernel_size=500,
#         mask_gap=1,
#         epoch_max=10000,
#         y_ticks=[0.05 + 0.05 * item for item in range(4)],
#         ylim=[0.02, 0.23],
#         y_ticks_format="${%.2f}$",
#         ncol=3,
#         timestring=time_string,
#         all_flag=all_flag,
#     )
#     draw_paper_figure_best_turing(
#         timestring_dict={
#             "PINN": ["20230113_012543_502301", "20230113_152941_190182", "20230114_050659_463810"],
#             "FNN": ["20230113_012543_388054", "20230113_020130_943114", "20230113_023725_022232",
#                     "20230113_031314_069539", "20230113_034902_802106", "20230113_042449_490497",
#                     "20230113_050033_762260", "20230113_053627_898635", "20230113_061223_021126",
#                     "20230113_064812_984827"],
#             "SB-FNN": ["20230113_044715_851325", "20230113_054236_639695", "20230113_063806_446492",
#                           "20230113_073328_129274", "20230113_082852_515273", "20230113_092420_414350",
#                           "20230113_101949_559684", "20230113_111508_182022", "20230113_121028_140717",
#                           "20230113_130547_991195"],#(A)
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
#                 model_name_short),#(A)
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=1000,
#         fontsize=30,
#         timestring=time_string,
#         all_flag=all_flag,
#     )
#
# def one_time_plot_rep(time_string=None, all_flag=False):
#     from model_REP_Lambda import Config
#     model_name_short = "REP"
#     time_string = get_now_string() if not time_string else time_string
#     draw_paper_figure_loss(
#         timestring_dict={
#             "PINN": ["20230113_072355_474015", "20230113_075729_957285", "20230113_083048_636737", "20230113_090445_351975", "20230113_093935_520438", "20230113_101402_523720", "20230113_104742_611608", "20230113_112205_274969", "20230113_115755_624347", "20230113_123231_431485"],
#             "FNN": ["20230113_075739_832183", "20230113_083929_235082", "20230113_092123_837192", "20230113_100401_481913", "20230113_104713_061893", "20230113_112931_198531", "20230113_121254_870794", "20230113_125559_423198", "20230113_134041_313176", "20230113_142505_384061"],
#             # "SB-FNN(A)": ["20230113_104739_849772", "20230113_114307_077581", "20230113_123924_947628", "20230113_133439_053026", "20230113_143009_620944", "20230113_152528_950426", "20230113_162054_882916", "20230113_171632_140936", "20230113_181200_480415", "20230113_190758_193187"],
#             # "SB-FNN(P)": ["20230113_085528_076951", "20230113_094201_864914", "20230113_102937_708564", "20230113_111531_049110", "20230113_120023_126623", "20230113_124608_611381", "20230113_133212_229659", "20230113_141806_950491", "20230113_150305_232567", "20230113_154838_167886"],
#             "SB-FNN": ["20230113_130616_626987", "20230113_140217_611121", "20230113_145746_666743", "20230113_155742_700314", "20230113_165401_732762", "20230113_175040_795192", "20230113_184641_305244", "20230113_194322_514807", "20230113_203951_160971", "20230113_213559_235176"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             # "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             # "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         kernel_size=2000,
#         mask_gap=1,
#         epoch_max=50000,
#         y_ticks=[-5 + 1 * item for item in range(9)],
#         ylim=[-5.5, 3.5],
#         y_ticks_format="$10^{%d}$",
#         ncol=5,
#         timestring=time_string,
#         all_flag=all_flag,
#     )
#     draw_paper_figure_best(
#         timestring_dict={
#             "PINN": ["20230113_072355_474015", "20230113_075729_957285", "20230113_083048_636737", "20230113_090445_351975", "20230113_093935_520438", "20230113_101402_523720", "20230113_104742_611608", "20230113_112205_274969", "20230113_115755_624347", "20230113_123231_431485"],
#             "FNN": ["20230113_075739_832183", "20230113_083929_235082", "20230113_092123_837192", "20230113_100401_481913", "20230113_104713_061893", "20230113_112931_198531", "20230113_121254_870794", "20230113_125559_423198", "20230113_134041_313176", "20230113_142505_384061"],
#             # "SB-FNN(A)": ["20230113_104739_849772", "20230113_114307_077581", "20230113_123924_947628", "20230113_133439_053026", "20230113_143009_620944", "20230113_152528_950426", "20230113_162054_882916", "20230113_171632_140936", "20230113_181200_480415", "20230113_190758_193187"],
#             # "SB-FNN(P)": ["20230113_085528_076951", "20230113_094201_864914", "20230113_102937_708564", "20230113_111531_049110", "20230113_120023_126623", "20230113_124608_611381", "20230113_133212_229659", "20230113_141806_950491", "20230113_150305_232567", "20230113_154838_167886"],
#             "SB-FNN": ["20230113_130616_626987", "20230113_140217_611121", "20230113_145746_666743", "20230113_155742_700314", "20230113_165401_732762", "20230113_175040_795192", "20230113_184641_305244", "20230113_194322_514807", "20230113_203951_160971", "20230113_213559_235176"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             # "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             # "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=5000,
#         fontsize=30,
#         timestring=time_string,
#         show_legend=["${}$".format(item) for item in ["\hat{M}_{lacI}", "\hat{M}_{tetR}", "\hat{M}_{cI}", "\hat{P}_{cI}", "\hat{P}_{lacI}", "\hat{P}_{tetR}"]] + ["${}$".format(item) for item in ["M_{lacI}", "M_{tetR}", "M_{cI}", "P_{cI}", "P_{lacI}", "P_{tetR}"]],
#         all_flag=all_flag,
#     )
#
# def one_time_plot_cc1(time_string=None, all_flag=False):
#     from model_CC1_Lambda import Config
#     model_name_short = "CC1"
#     time_string = get_now_string() if not time_string else time_string
#     draw_paper_figure_loss(
#         timestring_dict={
#             "PINN": ["20230114_104738_082733", "20230114_110633_738392", "20230114_112534_011307", "20230114_114419_776550", "20230114_120321_408292", "20230114_122211_757770", "20230114_124112_930914", "20230114_125956_718534", "20230114_131928_319098", "20230114_133828_281110", "20230114_105828_462987", "20230114_111836_784358", "20230114_113847_415229", "20230114_115906_425832", "20230114_121916_339738", "20230114_123910_001294", "20230114_130014_384850", "20230114_132027_156401", "20230114_134018_860625", "20230114_140047_470748"],
#             "FNN": ["20230113_150114_122814", "20230113_170207_774184", "20230113_173218_621612", "20230113_180213_136149", "20230113_153824_742465", "20230113_174035_034880", "20230113_181000_331963", "20230113_184102_918462", "20230113_191619_207010", "20230113_194800_328890", "20230113_190234_884004", "20230113_200256_223610", "20230113_210323_501140", "20230113_213341_501826", "20230113_233409_823585"],
#             # "SB-FNN(A)": ["20230114_042652_455355", "20230114_051054_533232", "20230114_014100_517379", "20230114_043138_706828", "20230114_072151_686295", "20230114_045400_054312", "20230114_053845_970724", "20230114_070401_975744", "20230114_035946_702151", "20230114_044917_964045", "20230114_062205_601996", "20230114_070728_045072", "20230114_083957_451761", "20230114_101241_352682"],
#             # "SB-FNN(P)": ["20230113_200410_615541", "20230113_210247_686231", "20230113_230054_282851", "20230113_233021_933410", "20230113_235948_661313", "20230113_204321_910480", "20230113_224559_812952", "20230113_231609_139457", "20230113_234559_811241", "20230114_003354_004737", "20230114_010603_573678", "20230113_223126_956416", "20230113_233159_014195", "20230114_003102_913432", "20230114_010019_176617", "20230114_030006_786992"],
#             "SB-FNN": ["20230114_075356_293959", "20230114_083750_174536", "20230114_080525_254861", "20230114_110112_844998", "20230114_131356_291274", "20230114_135752_431809", "20230114_084711_173631", "20230114_113744_155337", "20230114_122001_852017", "20230114_134448_521618", "20230114_091341_089367", "20230114_095622_967548", "20230114_112156_646795", "20230114_120428_577061", "20230114_133048_012766", "20230114_145713_453669"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             # "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             # "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         kernel_size=2000,
#         mask_gap=1,
#         epoch_max=50000,
#         y_ticks=[-6 + 1 * item for item in range(9)],
#         ylim=[-6.5, 2.5],
#         y_ticks_format="$10^{%d}$",
#         ncol=5,
#         timestring=time_string,
#         all_flag=all_flag,
#     )
#     draw_paper_figure_best(
#         timestring_dict={
#             "PINN": ["20230114_104738_082733", "20230114_110633_738392", "20230114_112534_011307", "20230114_114419_776550", "20230114_120321_408292", "20230114_122211_757770", "20230114_124112_930914", "20230114_125956_718534", "20230114_131928_319098", "20230114_133828_281110", "20230114_105828_462987", "20230114_111836_784358", "20230114_113847_415229", "20230114_115906_425832", "20230114_121916_339738", "20230114_123910_001294", "20230114_130014_384850", "20230114_132027_156401", "20230114_134018_860625", "20230114_140047_470748"],
#             "FNN": ["20230113_150114_122814", "20230113_170207_774184", "20230113_173218_621612", "20230113_180213_136149", "20230113_153824_742465", "20230113_174035_034880", "20230113_181000_331963", "20230113_184102_918462", "20230113_191619_207010", "20230113_194800_328890", "20230113_190234_884004", "20230113_200256_223610", "20230113_210323_501140", "20230113_213341_501826", "20230113_233409_823585"],
#             # "SB-FNN(A)": ["20230114_042652_455355", "20230114_051054_533232", "20230114_014100_517379", "20230114_043138_706828", "20230114_072151_686295", "20230114_045400_054312", "20230114_053845_970724", "20230114_070401_975744", "20230114_035946_702151", "20230114_044917_964045", "20230114_062205_601996", "20230114_070728_045072", "20230114_083957_451761", "20230114_101241_352682"],
#             # "SB-FNN(P)": ["20230113_200410_615541", "20230113_210247_686231", "20230113_230054_282851", "20230113_233021_933410", "20230113_235948_661313", "20230113_204321_910480", "20230113_224559_812952", "20230113_231609_139457", "20230113_234559_811241", "20230114_003354_004737", "20230114_010603_573678", "20230113_223126_956416", "20230113_233159_014195", "20230114_003102_913432", "20230114_010019_176617", "20230114_030006_786992"],
#             "SB-FNN": ["20230114_075356_293959", "20230114_083750_174536", "20230114_080525_254861", "20230114_110112_844998", "20230114_131356_291274", "20230114_135752_431809", "20230114_084711_173631", "20230114_113744_155337", "20230114_122001_852017", "20230114_134448_521618", "20230114_091341_089367", "20230114_095622_967548", "20230114_112156_646795", "20230114_120428_577061", "20230114_133048_012766", "20230114_145713_453669"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             # "SB-FNN(A)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
#             #     model_name_short),
#             # "SB-FNN(P)": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
#             #     model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
#                 model_name_short),
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=1000,
#         fontsize=30,
#         timestring=time_string,
#         show_legend=["${}$".format(item) for item in ["\hat{MPF}", "\hat{Kin}_{p}", "\hat{APC}_{p}"]] + ["${}$".format(item) for item in ["MPF", "Kin_{p}", "APC_{p}"]],
#         all_flag=all_flag,
#     )
#
# def one_time_plot_toggle(time_string=None, all_flag=False):
#     from model_Toggle_Lambda import Config
#     model_name_short = "Toggle"
#     time_string = get_now_string() if not time_string else time_string
#     draw_paper_figure_loss(
#         timestring_dict={
#             "PINN": ["20230119_034841_142848", "20230119_035236_808768", "20230119_035632_326111", "20230119_040028_018782", "20230119_040425_385766", "20230119_040822_827089", "20230119_041221_250790", "20230119_041622_306062", "20230119_042022_633950", "20230119_042421_336993"],
#             "FNN": ["20230119_021041_122577", "20230119_022039_928504", "20230119_023138_652196", "20230119_024153_862204", "20230119_025237_115030", "20230119_030255_928889", "20230119_031305_677722", "20230119_032331_940361", "20230119_033349_200106", "20230119_034351_838542"],
#             "SB-FNN": ["20230119_021042_109471", "20230119_022026_342089", "20230119_023015_781722", "20230119_024000_374886", "20230119_024954_488754", "20230119_025949_649798", "20230119_030934_812575", "20230119_031920_559026", "20230119_032859_818037", "20230119_033847_561583"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#         },
#         model_name_short=model_name_short,
#         kernel_size=1000,
#         mask_gap=1,
#         epoch_max=11000,
#         y_ticks=[-9.0 + 2 * item for item in range(5)],
#         ylim=[-10, 0],
#         y_ticks_format="$10^{%d}$",
#         ncol=3,
#         timestring=time_string,
#         all_flag=all_flag,
#     )
#     draw_paper_figure_best(
#         timestring_dict={
#             "PINN": ["20230119_034841_142848", "20230119_035236_808768", "20230119_035632_326111", "20230119_040028_018782", "20230119_040425_385766", "20230119_040822_827089", "20230119_041221_250790", "20230119_041622_306062", "20230119_042022_633950", "20230119_042421_336993"],
#             "FNN": ["20230119_021041_122577", "20230119_022039_928504", "20230119_023138_652196", "20230119_024153_862204", "20230119_025237_115030", "20230119_030255_928889", "20230119_031305_677722", "20230119_032331_940361", "20230119_033349_200106", "20230119_034351_838542"],
#             "SB-FNN": ["20230119_021042_109471", "20230119_022026_342089", "20230119_023015_781722", "20230119_024000_374886", "20230119_024954_488754", "20230119_025949_649798", "20230119_030934_812575", "20230119_031920_559026", "20230119_032859_818037", "20230119_033847_561583"],
#         },
#         info_path_format_dict={
#             "PINN": "./saves/train/{0}_PINN_Lambda_{{0}}/{0}_PINN_Lambda_{{0}}_info.npy".format(model_name_short),
#             "FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(model_name_short),
#             "SB-FNN": "./saves/train/{0}_Fourier_Lambda_{{0}}/{0}_Fourier_Lambda_{{0}}_info.npy".format(
#                 model_name_short),
#         },
#         model_name_short=model_name_short,
#         config=Config(),
#         loss_average_length=1000,
#         fontsize=30,
#         timestring=time_string,
#         show_legend=["$\hat{{{}}}$".format(item) for item in ["U", "V"]] + ["${}$".format(item) for item in ["U", "V"]],
#         all_flag=all_flag,
#     )

def one_time_plot_rep3(time_string=None, all_flag=False):
    from model_REP3_Omega import Config
    model_name_short = "REP3"
    time_string = get_now_string() if not time_string else time_string
    timestring_dict = {
        # # "SB-FNN (adaptive6-0.001)": ["20230128_011324_123287", "20230128_024550_162651", "20230128_041753_489833"],
        # # "SB-FNN (adaptive6-0.003)": ["20230128_011327_019707", "20230128_020711_591318", "20230128_030130_257367"],
        # # "SB-FNN (adaptive6-0.005)": ["20230128_011327_071545", "20230128_020748_568768", "20230128_030138_384136"],
        # # "SB-FNN (adaptive6-0.01)": ["20230128_011326_934486", "20230128_020701_931134", "20230128_030009_951016"],
        # # "SB-FNN (adaptive5-0.001)": ["20230131_164750_558922", "20230131_181427_283639", "20230131_194059_774767",
        # #                              "20230131_211051_300966", "20230131_223759_565674"],
        # "SB-FNN (adaptive5-0.003)": ["", "", "20230131_182930_515704",
        #                              "20230131_192028_822094", "20230131_201138_197869"],
        # # "SB-FNN (adaptive5-0.005)": ["20230131_164753_168442", "20230131_173909_942389", "20230131_183029_586248",
        # #                              "20230131_192145_937197", "20230131_201253_436408"],
        # # "SB-FNN (adaptive5-0.01)": ["20230131_164753_069857", "20230131_173814_528255", "20230131_182856_225513",
        # #                             "20230131_191848_055274", "20230131_200907_519166"],
        #
        # "SB-FNN (elu)": ["20230128_043559_038620", "20230128_051426_704280", "20230128_055312_128973"],
        # "SB-FNN (gelu)": ["20230128_040824_447494", "20230128_044611_723213", "20230128_052428_207371"],
        # "SB-FNN (relu)": ["20230128_054635_419499", "20230128_065624_260222", "20230128_080303_324199"],
        # "SB-FNN (sin)": ["20230128_044224_711939", "20230128_052346_476131", ""],
        # "SB-FNN (softplus)": ["", "20230128_052055_672588", "20230128_060037_618186"],
        # "SB-FNN (tanh)": ["", "20230128_052222_422111", "20230128_060132_175007"],
        # "PINN": ["20230128_132415_707985", "20230128_135000_392929", "20230128_141554_433679"],

        #v0131-ablation
        "SB-FNN (gelu)": ["20230128_040824_447494", "20230128_044611_723213", "20230128_052428_207371",
                          "20230130_185949_556191", "20230130_194009_456974", "20230130_202049_747812",
                          "20230130_210137_730567", "20230130_214157_850237", "20230130_222216_809292",
                          "20230130_230325_481540"],
        "SB-FNN (gelu+boundary)": ["20230129_150002_419674", "20230129_154035_439917", "20230129_162034_395080",
                                   "20230130_185947_398502", "20230130_201126_140762", "20230130_212220_429313",
                                   "20230130_223301_587141", "20230130_234225_531951", "",
                                   "20230131_020148_484337"],
        "SB-FNN (gelu+cyclic)": ["20230129_153951_486804", "20230129_162015_334574", "",
                                 "20230130_194035_088247", "20230130_202051_103568", "",
                                 "20230130_214110_257371", "20230130_222107_977432", ""],

        # # "PINN": ["20230125_073943_141220", "20230125_001001_585981"],
        # # "SB-FNN (adaptive_6)": ["20230126_031827_311142", "20230126_040929_770175", "20230126_031826_378280"],
        # # "SB-FNN (adaptive_3)": ["20230126_031827_221980", "20230126_040255_641576", "20230126_031826_642766"],
        # # "SB-FNN (elu)": ["20230125_003448_020523", "20230125_011256_355110", "20230125_082443_238331"],
        # "SB-FNN (gelu)": ["20230124_234504_014392", "20230125_002145_365414", "20230125_081920_468925"],
        # # "SB-FNN (relu)": ["20230124_234503_844986", "20230125_002050_961190", "20230125_082110_602339"],
        # # "SB-FNN (sin)": ["20230125_005826_100493", "20230125_013603_631118", "20230125_085607_079119"],
        # # "SB-FNN (softplus)": ["20230125_012538_615411", "20230125_020332_903522", "20230125_085620_701421"],
        # # "SB-FNN (tanh)": ["20230125_005540_802196", "20230125_013132_572485", "20230125_083013_080349"],
        # "SB-FNN (gelu+boundary)": ["20230125_015108_174467", "20230125_022852_847120", "20230125_090253_249386"],
        # "SB-FNN (gelu+boundary_tense)": ["20230127_004115_321527", "20230127_011943_050618", "20230127_015809_930792"],
        # "SB-FNN (gelu+cyclic-1)": ["20230126_192423_092041", "20230126_200218_223136", "20230126_204208_313040"],
        # "SB-FNN (gelu+cyclic-2)": ["20230127_020154_680831", "20230127_024510_758013", "20230127_032935_543922"],

        # "PINN": ["20230123_051031_562368", "20230123_053357_980705", "20230123_055749_996809", "20230123_062146_202857", "20230123_064557_873001", "20230123_070943_333342", "20230123_073321_157872", "20230123_075659_336315", "20230123_082045_071531", "20230123_084406_654492"],
        # "SB-FNN (adaptive)": ["20230123_183336_929195", "20230123_194400_864789", "20230123_205522_803460", "20230123_220609_712838", "20230123_231501_548374", "20230124_002412_546527", "20230124_013228_072621", "20230124_024156_434477", "20230124_035139_118738", "20230124_050237_890136"],
        # "SB-FNN (elu)": ["20230122_172356_141807", "20230122_175235_141685", "20230122_182111_620711", "20230122_185005_485841", "20230122_191839_797910", "20230122_194706_555950", "20230122_201542_275727", "20230122_204426_078612", "20230122_211302_537086", "20230122_214143_942954"],
        # "SB-FNN (gelu)": ["20230122_172354_858171", "20230122_181247_540053", "20230122_190131_809197", "20230122_195025_413886", "20230122_204004_116694", "20230122_213023_366107", "20230122_222110_724377", "20230122_231141_275867", "20230123_000057_919017", "20230123_004910_992568"],
        # "SB-FNN (relu)": ["20230122_172355_674477", "20230122_175301_358237", "20230122_182214_287156", "20230122_185121_852161", "20230122_192020_365122", "20230122_194927_741613", "20230122_201836_283604", "20230122_204730_847692", "20230122_211630_625231", "20230122_214532_931362"],
        # "SB-FNN (sin)": ["20230122_172355_913631", "20230122_181955_150804", "20230122_184811_964315", "20230122_191629_153172", "20230122_194444_112264", "20230122_201230_178897", "20230122_204109_459102", "20230122_210858_581205"],
        # "SB-FNN (softplus)": ["20230122_213726_263570", "20230122_220522_400243", "20230122_223255_024858", "20230122_230022_359009", "20230122_232755_645488", "20230122_235542_132621", "20230123_002354_817482", "20230123_005133_839592", "20230123_011851_309298", "20230123_014634_383504"],
        # "SB-FNN (tanh)": ["20230122_172356_062564", "20230122_175224_293261", "20230122_182113_947643", "20230122_184950_442320", "20230122_191830_278756", "20230122_194703_803581", "20230122_201526_928763", "20230122_204357_546386", "20230122_211235_173693", "20230122_214057_188257"],
    }
    draw_paper_figure_loss(
        timestring_dict=timestring_dict,
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
            "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=500,
        mask_gap=1,
        epoch_max=50000,
        y_ticks=[-6.0 + 1 * item for item in range(6)],
        ylim=[-6.5, 0.5],
        y_ticks_format="$10^{%d}$",
        ncol=2,
        legend_fontsize=18,
        timestring=time_string,
        all_flag=all_flag,
    )
    # draw_paper_figure_best(
    #     timestring_dict=timestring_dict,
    #     info_path_format_dict={
    #         "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
    #         "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
    #     },
    #     model_name_short=model_name_short,
    #     config=Config(),
    #     loss_average_length=5000,
    #     fontsize=30,
    #     timestring=time_string,
    #     show_legend=["${}$".format(item) for item in ["\hat{P}_{cI}", "\hat{P}_{lacI}", "\hat{P}_{tetR}"]] + ["${}$".format(item) for item in ["P_{cI}", "P_{lacI}", "P_{tetR}"]],
    #     all_flag=all_flag,
    # )

def one_time_plot_rep6(time_string=None, all_flag=False):
    from model_REP6_Omega import Config
    model_name_short = "REP6"
    time_string = get_now_string() if not time_string else time_string
    timestring_dict = {

        # "SB-FNN (adaptive3-0.001)": ["20230126_031826_502175", "20230126_041539_894499", "20230126_031826_895026"],
        # "SB-FNN (elu)": ["20230124_194554_708493", "20230124_203537_282673", "20230125_094055_807396"],
        # "SB-FNN (gelu)": ["20230124_194554_891475", "20230124_203949_776226", "20230125_093348_098237"],
        # "SB-FNN (relu)": ["20230124_194554_849562", "20230124_203722_950773", "20230125_094029_526171"],
        # "SB-FNN (sin)": ["20230124_212931_190163", "20230124_222209_980054", "20230125_103259_437037"],
        # "SB-FNN (softplus)": ["20230124_213144_456506", "20230124_222324_644725", "20230125_103338_369995"],
        # "SB-FNN (tanh)": ["20230124_212409_470578", "20230124_221538_636326", "20230125_102532_817191"],
        # "PINN": ["20230125_090736_256184", "20230125_095513_339197"],

        "SB-FNN (gelu)": ["20230124_194554_891475", "20230124_203949_776226", "20230125_093348_098237"],
        "SB-FNN (gelu+boundary)": ["20230127_004106_883307", "20230127_013654_085794", "20230127_023348_043369"],
        "SB-FNN (gelu+cyclic)": ["", "20230127_020156_745731", "20230127_030444_804640", "20230127_040657_389688"],


        # "SB-FNN (adaptive6-0.001)": ["20230128_011326_960261", "20230128_022236_204478", "20230128_033209_939203"],
        # "SB-FNN (adaptive6-0.003)": ["20230128_011326_961121", "20230128_022203_365466", "20230128_033140_201876"],
        # "SB-FNN (adaptive6-0.005)": ["20230128_011326_961948", "20230128_022221_825449", "20230128_033204_885332"],
        # "SB-FNN (adaptive6-0.01)": ["20230128_011326_617208", "20230128_022210_036101", "20230128_032904_107927"],
        # "SB-FNN (adaptive5-0.001)": ["20230131_164753_113986", "20230131_175400_952937", "20230131_190045_085207",
        #                              "20230131_200717_508042", "20230131_211350_305148"],
        # "SB-FNN (adaptive5-0.003)": ["20230131_164753_077559", "20230131_175354_683449", "20230131_190020_166909",
        #                              "20230131_200717_483959", "20230131_211305_292488"],
        # "SB-FNN (adaptive5-0.005)": ["20230131_164753_102901", "20230131_175447_202670", "20230131_190201_012982",
        #                              "20230131_200851_628570", "20230131_211531_604976"],
        # "SB-FNN (adaptive5-0.01)": ["20230131_164752_871211", "20230131_175142_860040", "20230131_185519_348155",
        #                             "20230131_195909_900507", "20230131_210411_992248"],
        # "SB-FNN (elu)": ["20230128_063152_257332", "20230128_072619_167239", "20230128_082046_220601"],
        # "SB-FNN (gelu)": ["20230128_060203_658882", "20230128_065608_271045", "20230128_075132_332564"],
        # "SB-FNN (relu)": ["20230128_070135_991237", "20230128_075935_588031", "20230128_085653_434705"],
        # "SB-FNN (sin)": ["20230128_064227_037698", "20230128_073849_170759", "20230128_083647_808956"],
        # "SB-FNN (softplus)": ["20230128_064018_738049", "20230128_073657_914287", "20230128_083402_903283"],
        # "SB-FNN (tanh)": ["20230128_064704_852360", "20230128_074343_227242", "20230128_084017_253247"],
        # "PINN": ["20230128_142357_492840", "20230128_151605_383892", "20230128_160838_098983"],

        # "SB-FNN (gelu)": ["20230128_065608_271045", "20230128_075132_332564", "20230130_185948_982141", "20230130_195616_636482", "20230130_205256_196309", "20230130_214915_074455", "20230130_224653_890767", "20230130_234536_075293", "20230131_004220_443351"],
        # "SB-FNN (gelu+boundary)": ["20230129_165341_618691", "20230130_185949_482964", "20230130_195629_438893", "20230130_205309_830545", "20230130_214921_516748", "20230130_224627_995486", "20230131_004037_887937"],
        # "SB-FNN (gelu+cyclic)": ["", "20230129_155545_087400", "20230129_165126_279015", "20230130_185949_031953", "20230130_195531_391140", "20230130_205216_050502", "20230130_224344_551905", "", ""],


        # # "PINN": ["20230125_090736_256184", "20230125_095513_339197"],
        # # "SB-FNN (adaptive_6)": ["20230126_031826_428476", "20230126_042115_785754", "20230126_031826_650297"],
        # # "SB-FNN (adaptive_3)": ["20230126_031826_502175", "20230126_041539_894499", "20230126_031826_895026"],
        # # "SB-FNN (elu)": ["20230124_194554_708493", "20230124_203537_282673", "20230125_094055_807396"],
        # "SB-FNN (gelu)": ["20230124_194554_891475", "20230124_203949_776226", "20230125_093348_098237"],
        # # "SB-FNN (relu)": ["20230124_194554_849562", "20230124_203722_950773", "20230125_094029_526171"],
        # # "SB-FNN (sin)": ["20230124_212931_190163", "20230124_222209_980054", "20230125_103259_437037"],
        # # "SB-FNN (softplus)": ["20230124_213144_456506", "20230124_222324_644725", "20230125_103338_369995"],
        # # "SB-FNN (tanh)": ["20230124_212409_470578", "20230124_221538_636326", "20230125_102532_817191"],
        # "SB-FNN (gelu+boundary)": ["20230124_234502_821689", "20230125_011421_453306", "20230125_104243_389203"],
        # "SB-FNN (gelu+boundary_tense)": ["20230127_004106_883307", "20230127_013654_085794", "20230127_023348_043369"],
        # "SB-FNN (gelu+cyclic-1)": ["20230126_191154_455798", "20230126_204807_161086"],
        # "SB-FNN (gelu+cyclic-2)": ["20230127_020156_745731", "20230127_030444_804640", "20230127_040657_389688"],

        # "PINN": ["20230123_051901_465405", "20230123_060508_057827", "20230123_065048_665629", "20230123_073721_504787", "20230123_082310_969669", "20230123_090946_428982", "20230123_095703_645997", "20230123_104341_700825", "20230123_113048_662712", "20230123_121750_730901"],
        # "SB-FNN (adaptive)": ["20230123_183343_390251", "20230123_192902_174778", "20230123_202406_750541", "20230123_211826_111533", "20230123_221311_998306", "20230123_230824_908542", "20230124_000346_570641", "20230124_005815_990397", "20230124_015352_491861", "20230124_024902_199353"],
        # "SB-FNN (elu)": ["20230122_221441_358430", "20230122_225657_744417", "20230122_233933_262407", "20230123_002159_398366", "20230123_010420_260520", "20230123_014730_009699", "20230123_022935_951136", "20230123_031148_471480", "20230123_035355_398537", "20230123_043643_167243"],
        # "SB-FNN (gelu)": ["20230122_220928_237797", "20230122_225109_559786", "20230122_233300_906604", "20230123_001510_629771", "20230123_005651_764535", "20230123_013919_995174", "20230123_022105_809396", "20230123_030254_601911", "20230123_034501_813513", "20230123_042724_145468"],
        # "SB-FNN (relu)": ["20230122_221024_568988", "20230122_225208_866366", "20230122_233348_376805", "20230123_001515_778693", "20230123_005746_059210", "20230123_013951_291585", "20230123_022131_947197", "20230123_030326_042313", "20230123_034514_920069", "20230123_042716_155492"],
        # "SB-FNN (sin)": ["20230123_021400_104350", "20230123_025441_712992", "20230123_033450_921820", "20230123_041535_510553", "20230123_053739_509237", "20230123_070000_473880", "20230123_074111_443724", "20230123_082152_460108"],
        # "SB-FNN (softplus)": ["20230123_050921_166396", "20230123_055137_763584", "20230123_063349_593738", "20230123_071534_550354", "20230123_075731_039961", "20230123_083926_896171", "20230123_092155_127385", "20230123_100355_525413", "20230123_104624_541203", "20230123_112848_450799"],
        # "SB-FNN (tanh)": ["20230123_013948_544409", "20230123_025052_147910", "20230123_040350_741139", "20230123_051539_107788", "20230123_062708_478036", "20230123_074054_020961", "20230123_085408_921989", "20230123_100453_573582", "20230123_111557_066662", "20230123_122942_723514"],
    }
    draw_paper_figure_loss(
        timestring_dict=timestring_dict,
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
            "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=2000,
        mask_gap=1,
        epoch_max=50000,
        y_ticks=[-6.0 + 1 * item for item in range(9)],
        ylim=[-6.5, 2.5],
        y_ticks_format="$10^{%d}$",
        ncol=2,
        legend_fontsize=18,
        timestring=time_string,
        all_flag=all_flag,
    )
    # draw_paper_figure_best(
    #     timestring_dict=timestring_dict,
    #     info_path_format_dict={
    #         "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
    #         "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
    #     },
    #     model_name_short=model_name_short,
    #     config=Config(),
    #     loss_average_length=5000,
    #     fontsize=30,
    #     timestring=time_string,
    #     show_legend=["${}$".format(item) for item in ["\hat{P}_{cI}", "\hat{P}_{lacI}", "\hat{P}_{tetR}"]] + ["${}$".format(item) for item in ["P_{cI}", "P_{lacI}", "P_{tetR}"]],
    #     all_flag=all_flag,
    # )

def one_time_plot_sir(time_string=None, all_flag=False):
    from model_SIR_Omega import Config
    model_name_short = "SIR"
    time_string = get_now_string() if not time_string else time_string
    timestring_dict = {
        # onetime - zjx
        "SB-FNN (gelu)": ["20230208_054923_480735", "20230208_055021_547672", "20230208_055103_605752", "20230208_055219_605214", "20230208_055200_731571"],
        "SB-FNN (gelu+boundary)": ["20230208_055814_242722", "20230208_055755_360284", "20230208_055618_221568", "20230208_055646_216024", "20230208_055705_623117"],
        "SB-FNN (gelu+stable)": ["20230208_065158_534423", "20230208_065213_586001", "20230208_065006_982076", "20230208_065033_016574", "20230208_065125_891344"],



        # # "SB-FNN (adaptive6-0.001)": ["20230128_011326_699415", "20230128_021151_384559", "20230128_030959_396809"],
        # # "SB-FNN (adaptive6-0.003)": ["20230128_035523_850497", "20230128_045841_486262", "20230128_060146_214353"],
        # # "SB-FNN (adaptive6-0.005)": ["20230128_035525_290377", "20230128_045819_964181", "20230128_060028_819657"],
        # # "SB-FNN (adaptive6-0.01)": ["20230128_035557_683120", "20230128_045943_877612", "20230128_060245_937317"],
        # # "SB-FNN (adaptive5-0.001)": ["20230131_164752_801634", "20230131_174438_374724", "20230131_184046_532334",
        # #                              "20230131_193708_389765", "20230131_203323_620516"],
        # # "SB-FNN (adaptive5-0.003)": ["20230131_205943_204225", "20230131_215758_674733", "20230131_225615_105475",
        # #                              "20230131_235354_134668", "20230201_005202_593906"],
        # "SB-FNN (adaptive5-0.005)": ["20230131_210316_463371", "20230131_220240_229268", "20230131_230129_205798",
        #                              "20230201_000031_882412", "20230201_005932_886872"],
        # # "SB-FNN (adaptive5-0.01)": ["20230131_210420_878534", "20230131_220446_963665", "20230131_230458_312790",
        # #                             "20230201_000424_978389", "20230201_010416_543746"],
        # "SB-FNN (elu)": ["20230128_092502_362573", "20230128_101241_019415", "20230128_105934_215096"],
        # "SB-FNN (gelu)": ["20230128_091516_720660", "20230128_100029_406401", "20230128_104552_276076"],
        # "SB-FNN (relu)": ["20230128_093616_917541", "20230128_102314_512683", "20230128_111020_808380"],
        # "SB-FNN (sin)": ["20230128_093507_139020", "20230128_102451_530060", "20230128_111501_664410"],
        # "SB-FNN (softplus)": ["20230128_093146_732741", "20230128_102020_230495", "20230128_110855_554563"],
        # "SB-FNN (tanh)": ["20230128_093559_955876", "20230128_102351_545916", "20230128_110941_623957"],
        # "PINN": ["20230128_143508_308883", "20230128_150313_980414", "20230128_153117_802370"],

        # "SB-FNN (gelu)": ["20230128_091516_720660", "20230128_100029_406401", "20230128_104552_276076",
        #                   "20230130_185949_434835", "20230130_194728_424445", "20230130_203533_142938",
        #                   "20230130_212239_690945", "20230130_221008_035357", "20230130_225744_487058",
        #                   "20230130_234550_867261"],
        # "SB-FNN (gelu+boundary_0.0)": ["20230129_150001_261616", "20230129_154420_031312", "20230129_162851_296638",
        #                            "20230130_185948_515124", "20230130_194437_749316", "20230130_202926_806121",
        #                            "20230130_211438_478704", "20230130_215934_301888", "20230130_224417_055163",
        #                            "20230130_232916_280818"],
        # # "SB-FNN (gelu+boundary_0.3)": ["20230202_023033_517692", "20230202_031447_656804", "20230202_035909_177246", "20230202_044358_444926", "20230202_052850_493587", "20230202_061314_764647", "20230202_065736_004268", "20230202_074146_961868", "20230202_082606_729977", "20230202_091019_238577"],
        #
        # "SB-FNN (gelu+stable)": ["20230129_150001_192396", "20230129_154741_905256", "20230129_163404_683891",
        #                          "20230130_185949_348764", "20230130_194545_470466", "20230130_203100_040895",
        #                          "20230130_211620_792766", "20230130_220157_980961", "20230130_224723_839633",
        #                          "20230130_233319_697327"],

        # # "PINN": ["20230125_043610_049048", "20230125_050126_860492"],
        # # "SB-FNN (adaptive_6)": ["20230126_031826_971572", "20230126_041230_136715", "20230126_040634_032885"],
        # # "SB-FNN (adaptive_3)": ["20230126_031826_556026", "20230126_040828_843979", "20230126_040828_843979"],
        # # "SB-FNN (elu)": ["20230125_025231_824805", "20230125_033410_274947"],
        # "SB-FNN (gelu)": ["20230125_024107_896191", "20230125_033710_764104"],
        # # "SB-FNN (relu)": ["20230125_024350_419106", "20230125_035611_080427"],
        # # "SB-FNN (sin)": ["20230125_040440_681453", "20230125_044727_745941"],
        # # "SB-FNN (softplus)": ["20230125_041514_287338", "20230125_045626_868292"],
        # # "SB-FNN (tanh)": ["20230125_030659_696879", "20230125_035128_503958"],
        # "SB-FNN (gelu+boundary)": ["20230125_042647_252543", "20230125_051459_076017"],
        # "SB-FNN (gelu+boundary_tense)": ["20230127_003859_334585", "20230127_012622_437093", "20230127_021353_520808"],
        # "SB-FNN (gelu+stable)": ["20230127_021010_370517", "20230127_025409_892965", "20230127_033816_777286"],

        # "PINN": ["20230123_234717_299531", "20230124_001058_364264", "20230124_003435_131193", "20230124_005814_160509", "20230124_012138_581311", "20230124_014501_414241", "20230124_020850_290943", "20230124_023254_346000", "20230124_025639_137893", "20230124_032006_207141"],
        # "SB-FNN (adaptive)": ["20230124_061025_725337", "20230124_072430_636371", "20230124_083802_866146", "20230124_095343_841909", "20230124_110829_018813", "20230124_122210_634107"],
        # "SB-FNN (elu)": ["20230124_034328_713901", "20230124_043511_564992", "20230124_051124_819139", "20230124_054922_921140", "20230124_064157_311244", "20230124_071838_849152", "20230124_075727_715078", "20230124_084848_472913", "20230124_092945_787858", "20230124_100913_525051"],
        # "SB-FNN (gelu)": ["20230123_234717_658744", "20230124_001804_811951", "20230124_004901_765421", "20230124_011954_018453", "20230124_015113_037197", "20230124_022152_955902", "20230124_025241_450137", "20230124_032337_906417", "20230124_035423_500518", "20230124_042512_572955"],
        # "SB-FNN (relu)": ["20230123_234717_521979", "20230124_001809_182519", "20230124_004904_538605", "20230124_012005_986120", "20230124_015044_951891", "20230124_022053_733167", "20230124_025101_207233", "20230124_032136_507930", "20230124_035216_114646", "20230124_042252_699990"],
        # "SB-FNN (sin)": ["20230124_045344_918684", "20230124_052521_966500", "20230124_055625_016518", "20230124_062724_069112", "20230124_065859_750690", "20230124_073010_426866", "20230124_080138_450202", "20230124_083301_875812", "20230124_090356_638837", "20230124_093502_318268"],
        # "SB-FNN (softplus)": ["20230124_045613_034502", "20230124_052715_129470", "20230124_055821_900577", "20230124_062918_758691", "20230124_070030_941960", "20230124_073144_283919", "20230124_080250_771649", "20230124_083351_546225", "20230124_090514_300894", "20230124_093615_638393"],
        # "SB-FNN (tanh)": ["20230124_034350_561187", "20230124_041647_284457", "20230124_044937_757848", "20230124_052228_418334", "20230124_055459_169706", "20230124_062733_955693", "20230124_065951_907837", "20230124_073213_492178", "20230124_080449_703917", "20230124_083732_771190"],
    }
    draw_paper_figure_loss(
        timestring_dict=timestring_dict,
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
            "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=5000,
        mask_gap=1,
        epoch_max=50000,
        y_ticks=[-9.0 + 2 * item for item in range(6-4)],
        ylim=[-9.5, 1.5-8],
        y_ticks_format="$10^{%d}$",
        ncol=2,
        legend_fontsize=18,
        timestring=time_string,
        all_flag=all_flag,
    )
    # draw_paper_figure_best(
    #     timestring_dict=timestring_dict,
    #     info_path_format_dict={
    #         "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
    #         "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
    #     },
    #     model_name_short=model_name_short,
    #     config=Config(),
    #     loss_average_length=5000,
    #     fontsize=30,
    #     timestring=time_string,
    #     show_legend=["${}$".format(item) for item in ["\hat{P}_{cI}", "\hat{P}_{lacI}", "\hat{P}_{tetR}"]] + ["${}$".format(item) for item in ["P_{cI}", "P_{lacI}", "P_{tetR}"]],
    #     all_flag=all_flag,
    # )

def one_time_plot_siraged(time_string=None, all_flag=False):
    from model_SIRAged_Omega import Config
    model_name_short = "SIRAged"
    time_string = get_now_string() if not time_string else time_string
    timestring_dict = {
        # "SB-FNN (adaptive6-0.001)": ["20230128_011326_697583", "20230128_035729_248775", "20230128_064107_455848"],
        # "SB-FNN (adaptive6-0.003)": ["20230128_011326_705516", "20230128_034916_299300", "20230128_062530_304271"],
        # "SB-FNN (adaptive6-0.005)": ["20230128_011326_548136", "20230128_035416_807507", "20230128_063426_256672"],
        # "SB-FNN (adaptive6-0.01)": ["20230128_011326_709593", "20230128_040013_951309", "20230128_064841_923259"],
        # "SB-FNN (adaptive5-0.001)": ["20230131_164752_938162", "20230131_192832_142281", "20230131_220930_754450",
        #                              "20230201_004859_938518", "20230201_033059_868656"],
        "SB-FNN (adaptive5-0.003)": ["20230131_164752_941184", "20230131_192209_296531", "20230131_215511_427133",
                                     "20230201_002758_020263", "20230201_025943_799657"],
        # "SB-FNN (adaptive5-0.005)": ["20230131_164752_795731", "20230131_192442_645006", "20230131_220117_608189",
        #                              "20230201_003722_604087", "20230201_031158_741267"],
        # "SB-FNN (adaptive5-0.01)": ["20230131_164752_024554", "20230131_193350_890739", "20230131_221256_530091",
        #                             "20230201_005552_673289", "20230201_033747_066292"],
        "SB-FNN (elu)": ["20230128_070558_945088", "20230128_093801_649410", "20230128_120942_202556"],
        "SB-FNN (gelu)": ["20230128_070418_503196", "20230128_093542_268374", "20230128_120455_802352"],
        "SB-FNN (relu)": ["20230128_091236_314042", "20230128_113752_476917", "20230128_140220_962851"],
        "SB-FNN (sin)": ["20230128_090203_328731", "20230128_112632_971170", "20230128_134946_160376"],
        "SB-FNN (softplus)": ["20230128_084613_159434", "20230128_111130_459877", "20230128_133509_082130"],
        "SB-FNN (tanh)": ["20230128_091157_356709", "20230128_132455_105304", "20230128_174038_972809"],
        "PINN": ["20230128_142935_605095", "20230128_171642_896138", "20230128_200057_375788"],

        # "SB-FNN (gelu)": ["20230128_070418_503196", "20230128_093542_268374", "20230128_120455_802352",
        #                   "20230130_185949_367867", "20230130_212908_371733", "20230130_235756_035880",
        #                   "20230131_022548_161700", "20230131_045302_972510", "20230131_071931_862538",
        #                   "20230131_094602_835989"],
        # "SB-FNN (gelu+boundary)": ["", "", "20230129_195937_952650",
        #                            "20230130_185949_362046", "20230130_212816_051930", "",
        #                            "", "20230131_044647_915757", "",
        #                            ""],
        # "SB-FNN (gelu+stable)": ["20230129_150002_340005", "20230129_172934_160364", "20230129_195736_754091",
        #                          "20230130_185949_388516", "20230130_212128_782413", "20230130_234300_368066",
        #                          "20230131_020533_840750", "20230131_042618_589137", "20230131_064703_249084",
        #                          "20230131_090807_119814"],

        # # "PINN": ["20230125_154408_578825", "20230125_180945_684235"],
        # # "SB-FNN (adaptive_6)": ["20230126_031827_196095", "20230126_054701_582730", "20230126_032500_176590"],
        # # "SB-FNN (adaptive_3)": ["20230126_031827_215554", "20230126_054017_250718", "20230126_040059_215999"],
        # # "SB-FNN (elu)": ["20230125_194912_660275", "20230125_233900_199719", "20230125_113513_249028"],
        # "SB-FNN (gelu)": ["20230125_160242_190270", "20230125_181935_084463", "20230125_112612_446592"],
        # # "SB-FNN (relu)": ["20230125_172100_002889", "20230125_194953_077806", "20230125_112640_653662"],
        # # "SB-FNN (sin)": ["20230125_203701_838486", "20230125_225457_237637", "20230125_134329_806003"],
        # # "SB-FNN (softplus)": ["20230125_204723_511250", "20230125_225922_839450", "20230125_134451_366695"],
        # # "SB-FNN (tanh)": ["20230125_203432_142685", "20230125_224345_685056", "20230125_133456_936209"],
        # "SB-FNN (gelu+boundary)": ["20230125_223801_813906", "20230126_005758_163026", "20230125_142334_670035"],
        # "SB-FNN (gelu+boundary_tense)": ["20230127_003854_169340", "20230127_045212_509992"],
        # "SB-FNN (gelu+stable)": ["20230127_021017_804031", "20230127_043458_050928", "20230127_005251_778418"],
    }
    draw_paper_figure_loss(
        timestring_dict=timestring_dict,
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
            "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=3000,
        mask_gap=1,
        epoch_max=50000,
        y_ticks=None,#[-9.0 + 2 * item for item in range(6)],
        ylim=None,#[-9.5, 1.5],
        y_ticks_format="$10^{%d}$",
        ncol=2,
        legend_fontsize=18,
        timestring=time_string,
        all_flag=all_flag,
    )
    # draw_paper_figure_best(
    #     timestring_dict=timestring_dict,
    #     info_path_format_dict={
    #         "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
    #         "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
    #     },
    #     model_name_short=model_name_short,
    #     config=Config(),
    #     loss_average_length=5000,
    #     fontsize=30,
    #     timestring=time_string,
    #     show_legend=["${}$".format(item) for item in ["\hat{P}_{cI}", "\hat{P}_{lacI}", "\hat{P}_{tetR}"]] + ["${}$".format(item) for item in ["P_{cI}", "P_{lacI}", "P_{tetR}"]],
    #     all_flag=all_flag,
    # )


def one_time_plot_turing1d(time_string=None, all_flag=False):
    from model_SIRAged_Omega import Config
    model_name_short = "Turing1D"
    time_string = get_now_string() if not time_string else time_string
    timestring_dict = {
        # "PINN": ["20230210_120822_659860", "20230210_130006_653333", "20230210_135258_098198", "20230210_071427_448774",
        #          "20230210_080844_597854"],

        # "SB-FNN (adaptive6-0.001)": ["20230210_053235_252677", "20230210_055449_182694", "20230210_061709_908943",
        #                              "20230210_090520_710998", "20230210_092738_336913"],
        # "SB-FNN (adaptive6-0.003)": ["20230210_053239_970816", "20230210_055459_628107", "20230210_061720_875008",
        #                              "20230210_093202_653412", "20230210_095432_917211"],
        # "SB-FNN (adaptive6-0.005)": ["20230210_053242_251538", "20230210_055503_531315", "20230210_061724_273860",
        #                              "20230210_093439_498185", "20230210_095639_026788"],
        # "SB-FNN (adaptive6-0.01)": ["20230210_053244_087759", "20230210_055511_930725", "20230210_061734_098059",
        #                             "20230210_095001_654763", "20230210_101218_812323"],
        # "SB-FNN (adaptive5-0.001)": ["20230210_053302_243597", "20230210_055332_390196", "20230210_061358_676147",
        #                              "20230210_101653_805491", "20230210_103730_422439"],
        # "SB-FNN (adaptive5-0.003)": ["20230210_053352_954182", "20230210_055412_926830", "20230210_061437_911260",
        #                              "20230210_101839_219881", "20230210_103852_428083"],
        # "SB-FNN (adaptive5-0.005)": ["20230210_063425_469481", "20230210_065453_141425", "20230210_071526_427995",
        #                              "20230210_103436_476600", "20230210_105506_495511"],
        "SB-FNN (gelu)": ["20230210_044747_189401", "20230210_050246_673677", "20230210_051742_263036", "20230210_073552_104710", "20230210_075035_185094"],
        "SB-FNN (gelu+boundary)": ["20230210_090206_908492", "20230210_091707_174602"],
        "SB-FNN (gelu+stable)": ["20230210_044818_467857", "20230210_050313_480222", "20230210_051805_905028",
                                 "20230210_090517_556835", "20230210_091957_838714"],
        # "SB-FNN (adaptive5-0.01)": ["20230210_063501_588006", "20230210_065516_749927", "20230210_071531_856634",
        #                             "20230210_105805_604766", "20230210_111841_145092"],
        # "SB-FNN (elu)": ["20230210_044747_166796", "20230210_050247_181748", "20230210_051743_433935",
        #                  "20230210_080516_280199", "20230210_081955_706792"],
        # "SB-FNN (relu)": ["20230210_044746_988832", "20230210_050246_717237", "20230210_051744_474600",
        #                   "20230210_073557_926196", "20230210_075051_971803"],
        # "SB-FNN (sin)": ["20230210_044746_350982", "20230210_050307_605571", "20230210_051829_291561",
        #                  "20230210_083440_675543", "20230210_084959_588235"],
        # "SB-FNN (softplus)": ["20230210_083532_920062", "20230210_085024_332654"],
        # "SB-FNN (tanh)": ["20230210_044746_345810", "20230210_050242_940678", "20230210_051738_175364",
        #                   "20230210_080546_947960"],

    }
    draw_paper_figure_loss(
        timestring_dict=timestring_dict,
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
            "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
        kernel_size=500,
        mask_gap=1,
        epoch_max=5000,
        y_ticks=[-5.0 + 2 * item for item in range(5)],
        ylim=[-5.5, 3.5],
        y_ticks_format="$10^{%d}$",
        ncol=2,
        legend_fontsize=18,
        timestring=time_string,
        all_flag=all_flag,
    )
    # draw_paper_figure_best(
    #     timestring_dict=timestring_dict,
    #     info_path_format_dict={
    #         "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
    #         "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
    #     },
    #     model_name_short=model_name_short,
    #     config=Config(),
    #     loss_average_length=5000,
    #     fontsize=30,
    #     timestring=time_string,
    #     show_legend=["${}$".format(item) for item in ["\hat{P}_{cI}", "\hat{P}_{lacI}", "\hat{P}_{tetR}"]] + ["${}$".format(item) for item in ["P_{cI}", "P_{lacI}", "P_{tetR}"]],
    #     all_flag=all_flag,
    # )

strings = {
    "PINN": """
20230210_120822_659860
20230210_130006_653333
20230210_135258_098198
20230210_071427_448774
20230210_080844_597854
    """,
    "SB-FNN (gelu)": """
20230210_044747_189401
20230210_050246_673677
20230210_051742_263036
20230210_073552_104710
20230210_075035_185094
    """,
    "SB-FNN (gelu+boundary)": """
20230210_090206_908492
20230210_091707_174602
    """,
    "SB-FNN (gelu+stable)": """
20230210_044818_467857
20230210_050313_480222
20230210_051805_905028
20230210_090517_556835
20230210_091957_838714
    """,
    "SB-FNN (adaptive6-0.001)": """
20230210_053235_252677
20230210_055449_182694
20230210_061709_908943
20230210_090520_710998
20230210_092738_336913
    """,
    "SB-FNN (adaptive6-0.003)": """
20230210_053239_970816
20230210_055459_628107
20230210_061720_875008
20230210_093202_653412
20230210_095432_917211
    """,
    "SB-FNN (adaptive6-0.005)": """
20230210_053242_251538
20230210_055503_531315
20230210_061724_273860
20230210_093439_498185
20230210_095639_026788
    """,
    "SB-FNN (adaptive6-0.01)": """
20230210_053244_087759
20230210_055511_930725
20230210_061734_098059
20230210_095001_654763
20230210_101218_812323
    """,
"SB-FNN (adaptive5-0.001)": """
20230210_053302_243597
20230210_055332_390196
20230210_061358_676147
20230210_101653_805491
20230210_103730_422439
    """,
    "SB-FNN (adaptive5-0.003)": """
20230210_053352_954182
20230210_055412_926830
20230210_061437_911260
20230210_101839_219881
20230210_103852_428083
    """,
    "SB-FNN (adaptive5-0.005)": """
20230210_063425_469481
20230210_065453_141425
20230210_071526_427995
20230210_103436_476600
20230210_105506_495511
    """,
    "SB-FNN (adaptive5-0.01)": """
20230210_063501_588006
20230210_065516_749927
20230210_071531_856634
20230210_105805_604766
20230210_111841_145092
    """,
    "SB-FNN (elu)": """
20230210_044747_166796
20230210_050247_181748
20230210_051743_433935
20230210_080516_280199
20230210_081955_706792
    """,
#     "SB-FNN (gelu)": """
#     20230128_070418_503196
# 20230128_093542_268374
# 20230128_120455_802352
#     """,
    "SB-FNN (relu)": """
20230210_044746_988832
20230210_050246_717237
20230210_051744_474600
20230210_073557_926196
20230210_075051_971803
    """,
    "SB-FNN (sin)": """
20230210_044746_350982
20230210_050307_605571
20230210_051829_291561
20230210_083440_675543
20230210_084959_588235
    """,
    "SB-FNN (softplus)": """
20230210_083532_920062
20230210_085024_332654
    """,
    "SB-FNN (tanh)": """
20230210_044746_345810
20230210_050242_940678
20230210_051738_175364
20230210_080546_947960
    """,
# "PINN": """
#     20230202_023033_517692
# 20230202_031447_656804
# 20230202_035909_177246
# 20230202_044358_444926
# 20230202_052850_493587
# 20230202_061314_764647
# 20230202_065736_004268
# 20230202_074146_961868
# 20230202_082606_729977
# 20230202_091019_238577
#     """,
}


if __name__ == "__main__":
    time_string = get_now_string()

    # one_time_draw_turing1d("20230207_152211_962959")
    # one_time_plot_rep3(time_string)
    # one_time_plot_rep6(time_string)
    # one_time_plot_sir(time_string)
    # one_time_plot_siraged(time_string)
    # clear_reformat_dictionary(strings)
    one_time_plot_turing1d(time_string)
    # one_time_plot_turing_best()
    # one_time_plot_sir_best()
    # one_time_plot_rep_best()
    # one_time_plot_cc1_best()

    # one_time_plot_cc1(time_string, all_flag=True)
    # one_time_plot_rep(time_string, all_flag=True)
    # one_time_plot_turing(time_string, all_flag=True)
    # one_time_plot_sir(time_string, all_flag=True)
    # one_time_plot_toggle(time_string, all_flag=False)

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
    # model_name_short = "REP3"
    # time_string = get_now_string() if not time_string else time_string
    # timestring_dict = {
    #     "SB-FNN (gelu+cyclic-2)": ["20230127_020154_680831"],
    # }
    # from model_REP3_Omega import Config
    # draw_paper_figure_best(
    #     timestring_dict=timestring_dict,
    #     info_path_format_dict={
    #         "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
    #         "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
    #     },
    #     model_name_short=model_name_short,
    #     config=Config(),
    #     loss_average_length=5000,
    #     fontsize=30,
    #     timestring=time_string,
    #     show_legend=["${}$".format(item) for item in ["\hat{P}_{cI}", "\hat{P}_{lacI}", "\hat{P}_{tetR}"]] + ["${}$".format(item) for item in ["P_{cI}", "P_{lacI}", "P_{tetR}"]],
    #     all_flag=False,
    # )
    # pass
