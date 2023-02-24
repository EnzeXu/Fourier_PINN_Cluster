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

def get_activation_weights(**kwargs):
    assert_keyword_list = ["timestring_dict", "info_path_format_dict", "model_name_short"]
    assert all(item in kwargs for item in assert_keyword_list)
    timestring_dict = kwargs["timestring_dict"]
    info_path_format_dict = kwargs["info_path_format_dict"]
    model_name_short = kwargs["model_name_short"]

    for i, one_model_group in enumerate(timestring_dict.keys()):
        timestring_list = timestring_dict[one_model_group]
        if one_model_group not in info_path_format_dict:
            info_path_format = info_path_format_dict["default"]
        else:
            info_path_format = info_path_format_dict[one_model_group]
        print(one_model_group)

        weights_sum = np.zeros(6)

        for one_timestring in timestring_list:
            if len(one_timestring) == 0:
                continue
            info_path = info_path_format.format(one_timestring)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            assert info["activation"] == "adaptive_6"
            weights_sum += info["activation_weights_record"][0, -1, :]
        print(weights_sum / len(timestring_list))







strings = {
    "PINN": """
20230211_060537_797828
20230211_065947_380670
20230211_075158_672358
20230211_084148_046944
20230211_093100_362313
    """,
    "SB-FNN (gelu)": """
20230211_060537_756402
20230211_062031_666596
20230211_063531_481589
20230211_065028_757798
20230211_070525_245486
    """,
    "SB-FNN (gelu+boundary)": """
20230211_064602_738746
20230211_070053_695387
20230211_071548_943831
20230211_073031_247541
20230211_074508_183149
    """,
    "SB-FNN (gelu+stable)": """
20230211_071953_563191
20230211_073428_487170
20230211_074901_798770
20230211_080336_311850
20230211_081806_466066
    """,
    "SB-FNN (adaptive6-0.001)": """
20230219_152513_324391
20230219_193109_906316
20230219_202729_158054
20230219_212348_636458
20230219_222006_137234
    """,
    "SB-FNN (adaptive6-0.003)": """
20230219_152704_181244
20230219_194037_387358
20230219_203702_250329
20230219_213323_033943
20230219_222942_217601
    """,
    "SB-FNN (adaptive6-0.005)": """
20230219_155940_752336
20230219_213249_389814
20230219_222912_621460
20230219_232534_796717
20230220_002156_493469
    """,
    "SB-FNN (adaptive6-0.01)": """
20230219_155954_756707
20230219_213341_205204
20230219_222959_839129
20230219_232619_434640
20230220_002236_686862
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
20230211_060537_111345
20230211_062040_256413
20230211_063542_703509
20230211_065045_221379
20230211_070546_083010
    """,
#     "SB-FNN (gelu)": """
#     20230128_070418_503196
# 20230128_093542_268374
# 20230128_120455_802352
#     """,
    "SB-FNN (relu)": """
20230211_060537_988802
20230211_062034_163633
20230211_063528_427728
20230211_065025_572317
20230211_070524_312738
    """,
    "SB-FNN (sin)": """
20230211_060537_064092
20230211_213808_075609
20230212_112915_697248
    """,
    "SB-FNN (softplus)": """
20230211_060537_016669
20230211_205131_548476
20230212_095528_981594
    """,
    "SB-FNN (tanh)": """
20230211_060537_195249
20230211_062029_636234
20230211_063522_031876
20230211_065011_524859
20230211_070501_047691
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
    timestring_dict = {
        "SB-FNN (adaptive6-0.001)": ["20230222_195425_504128", "20230223_042423_367311", "20230223_054835_624814",
                                     "20230223_071247_198508", "20230223_083701_347399"],
        "SB-FNN (adaptive6-0.003)": ["20230222_195734_705429", "20230223_044022_015999", "20230223_060436_486287", "20230223_072849_306519", "20230223_085302_153828"],

    }
    model_name_short = "Turing2D"
    get_activation_weights(
        timestring_dict=timestring_dict,
        info_path_format_dict={
            "PINN": "./saves/train/{0}_PINN_Omega_{{0}}/{0}_PINN_Omega_{{0}}_info.npy".format(model_name_short),
            "default": "./saves/train/{0}_Fourier_Omega_{{0}}/{0}_Fourier_Omega_{{0}}_info.npy".format(model_name_short),
        },
        model_name_short=model_name_short,
    )
    # one_time_draw_turing1d("20230207_152211_962959")
    # one_time_plot_rep3(time_string)
    # one_time_plot_rep6(time_string)
    # one_time_plot_sir(time_string)
    # one_time_plot_siraged(time_string)
    # clear_reformat_dictionary(strings)
    # one_time_plot_turing1d(time_string)
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
