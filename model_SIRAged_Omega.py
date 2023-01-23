import numpy as np
import time
import torch
import random
import pickle
import sys
import os

from torch import nn, optim, autograd
from scipy.integrate import odeint
import argparse

from utils import *


class Parameters:
    n = 5
    N = 100
    beta = 0.01
    gamma = 0.05
    M = np.asarray([
        [19.200, 4.800, 5.050, 3.400, 1.700],
        [4.800, 42.400, 5.900, 6.250, 1.733],
        [5.050, 5.900, 14.000, 7.575, 1.700],
        [3.400, 6.250, 7.575, 9.575, 1.544],
        [1.700, 1.733, 1.700, 1.544, 5.456],
    ])


class TrainArgs:
    iteration = 50000  # 20000 -> 50000
    epoch_step = 1000  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.01
    main_path = "."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config:
    def __init__(self):
        self.model_name = "SIRAged_Fourier_Omega"
        self.params = Parameters
        self.curve_names = ["S{}".format(i) for i in range(1, self.params.n + 1)] + ["I{}".format(i) for i in range(1, self.params.n + 1)] + ["R{}".format(i) for i in range(1, self.params.n + 1)]
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0
        self.layer = -1
        self.pinn = 0

        self.T = 100
        self.T_unit = 1e-2
        self.T_N = int(self.T / self.T_unit)

        self.prob_dim = len(self.curve_names)
        self.y0 = np.asarray([50.0] * self.params.n + [40.0] * self.params.n + [10.0] * self.params.n)
        self.boundary_list = np.asarray([[0.0, 100.0]] * 3 * self.params.n)
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        self.x = torch.tensor(np.asarray([[[i * self.T_unit] * 1 for i in range(self.T_N)]]),
                              dtype=torch.float32).to(self.device)
        self.truth = odeint(self.pend, self.y0, self.t)

        self.modes = 64  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.width = 16
        self.fc_map_dim = 128

        self.activation = ""
        self.cyclic = None
        self.stable = None
        self.boundary = None
        self.derivative = None
        self.skip_draw_flag = False
        self.loss_average_length = int(0.1 * self.args.iteration)

    def pend(self, y, t):
        k = self.params
        S_arr = y[0: k.n]
        I_arr = y[k.n: 2 * k.n]
        R_arr = y[2 * k.n: 3 * k.n]
        ds = []
        di = []
        dr = []
        for i in range(k.n):
            ds.append(- k.beta * S_arr[i] / k.N * sum([k.M[i][j] * I_arr[j] for j in range(k.n)]))
            di.append(k.beta * S_arr[i] / k.N * sum([k.M[i][j] * I_arr[j] for j in range(k.n)]) - k.gamma * I_arr[i])
            dr.append(k.gamma * I_arr[i])
        dydt = np.asarray(ds + di + dr)
        return dydt

# def penalty_func(x):
#     return 1 * (- torch.tanh((x - 1.5)) + 1)# return 1 * (- torch.tanh((x - 2.5)) + 1)


def penalty_func(x):
    return 1 * (- torch.tanh((x - 0.005) * 200) + 1) # 1 * (- torch.tanh((x - 0.004) * 300) + 1)


class FourierModel(nn.Module):
    def __init__(self, config):
        super(FourierModel, self).__init__()
        self.time_string = get_now_string()
        self.config = config
        self.setup_seed(self.config.seed)

        self.fc0 = nn.Linear(1, self.config.width)  # input channel is 2: (a(x), x)
        # self.layers = Layers(config=self.config, n=self.config.layer).to(self.config.device)
        self.conv1 = SpectralConv1d(self.config).to(self.config.device)
        self.conv2 = SpectralConv1d(self.config).to(self.config.device)
        self.conv3 = SpectralConv1d(self.config).to(self.config.device)
        self.conv4 = SpectralConv1d(self.config).to(self.config.device)
        self.cnn1 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.cnn2 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.cnn3 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.cnn4 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block3 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block4 = ActivationBlock(self.config).to(self.config.device)

        self.fc1 = nn.Linear(self.config.width, self.config.fc_map_dim)
        self.fc2 = nn.Linear(self.config.fc_map_dim, self.config.prob_dim)

        self.criterion = torch.nn.MSELoss().to(self.config.device)  # "sum"
        self.criterion_non_reduce = torch.nn.MSELoss(reduce=False).to(self.config.device)

        self.y_tmp = None
        self.epoch_tmp = None
        self.loss_record_tmp = None
        self.real_loss_mse_record_tmp = None
        self.real_loss_nmse_record_tmp = None
        self.time_record_tmp = None

        self.figure_save_path_folder = "{0}/saves/figure/{1}_{2}/".format(self.config.args.main_path,
                                                                          self.config.model_name, self.time_string)
        self.train_save_path_folder = "{0}/saves/train/{1}_{2}/".format(self.config.args.main_path,
                                                                        self.config.model_name, self.time_string)
        if not os.path.exists(self.figure_save_path_folder):
            os.makedirs(self.figure_save_path_folder)
        if not os.path.exists(self.train_save_path_folder):
            os.makedirs(self.train_save_path_folder)
        self.default_colors = ColorCandidate().get_color_list(self.config.prob_dim, 0.5)
        # self.default_colors = ["red", "blue", "green", "orange", "cyan", "purple", "pink", "indigo", "brown", "grey", "indigo", "olive"]

        myprint("using {}".format(str(self.config.device)), self.config.args.log_path)
        myprint("iteration = {}".format(self.config.args.iteration), self.config.args.log_path)
        myprint("epoch_step = {}".format(self.config.args.epoch_step), self.config.args.log_path)
        myprint("test_step = {}".format(self.config.args.test_step), self.config.args.log_path)
        myprint("model_name = {}".format(self.config.model_name), self.config.args.log_path)
        myprint("time_string = {}".format(self.time_string), self.config.args.log_path)
        myprint("seed = {}".format(self.config.seed), self.config.args.log_path)
        myprint("initial_lr = {}".format(self.config.args.initial_lr), self.config.args.log_path)
        myprint("cyclic = {}".format(self.config.cyclic), self.config.args.log_path)
        myprint("stable = {}".format(self.config.stable), self.config.args.log_path)
        myprint("derivative = {}".format(self.config.derivative), self.config.args.log_path)
        myprint("activation = {}".format(self.config.activation), self.config.args.log_path)
        # myprint("early stop: {}".format("On" if self.config.args.early_stop else "Off"), self.config.args.log_path)
        self.truth_loss()

    def truth_loss(self):
        y_truth = torch.tensor(self.config.truth.reshape([1, self.config.T_N, self.config.prob_dim])).to(
            self.config.device)
        tl, tl_list = self.loss(y_truth)
        loss_print_part = " ".join(
            ["Loss_{0:d}:{1:.8f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(tl_list)])
        myprint("Ground truth has loss: Loss:{0:.8f} {1}".format(tl.item(), loss_print_part), self.config.args.log_path)

    #  MSE-loss of predicted value against truth
    def real_loss(self, y):
        truth = torch.tensor(self.config.truth[:, 10:15]).to(self.config.device)
        real_loss_mse = self.criterion(y[0, :, 10:15], truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y[0, :, 10:15], truth) / (truth ** 2))
        return real_loss_mse, real_loss_nmse

    def early_stop(self):
        if not self.config.args.early_stop or len(self.loss_record_tmp) < 2 * self.config.args.early_stop_period:
            return False
        sum_old = sum(
            self.loss_record_tmp[- 2 * self.config.args.early_stop_period: - self.config.args.early_stop_period])
        sum_new = sum(self.loss_record_tmp[- self.config.args.early_stop_period:])
        if (sum_new - sum_old) / sum_old < - self.config.args.early_stop_tolerance:
            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.loss_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.loss_record_tmp) - self.config.args.early_stop_period,
                len(self.loss_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.args.log_path)
            return False
        else:
            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.loss_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.loss_record_tmp) - self.config.args.early_stop_period,
                len(self.loss_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.args.log_path)
            myprint("[Early Stop] Early Stop!", self.config.args.log_path)
            return True

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # x = self.layers(x)
        x1 = self.conv1(x)
        x2 = self.cnn1(x)
        x = x1 + x2
        x = self.activate_block1(x)

        x1 = self.conv2(x)
        x2 = self.cnn2(x)
        x = x1 + x2
        x = self.activate_block2(x)

        x1 = self.conv3(x)
        x2 = self.cnn3(x)
        x = x1 + x2
        x = self.activate_block3(x)

        x1 = self.conv4(x)
        x2 = self.cnn4(x)
        x = x1 + x2
        x = self.activate_block4(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def ode_gradient(self, x, y):
        k = self.config.params

        # U = y[0, :, 0]
        # V = y[0, :, 1]

        s = y[0, :, 0: k.n]
        i = y[0, :, k.n: k.n * 2]
        r = y[0, :, k.n * 2: k.n * 3]

        s_t_collection, i_t_collection, r_t_collection = [], [], []
        for ii in range(k.n):
            s_t_collection.append(
                torch.gradient(s[:, ii], spacing=(self.config.t_torch,))[0].reshape([self.config.T_N, 1]))
            i_t_collection.append(
                torch.gradient(i[:, ii], spacing=(self.config.t_torch,))[0].reshape([self.config.T_N, 1]))
            r_t_collection.append(
                torch.gradient(r[:, ii], spacing=(self.config.t_torch,))[0].reshape([self.config.T_N, 1]))
        s_t = torch.cat(s_t_collection, 1)
        i_t = torch.cat(i_t_collection, 1)
        r_t = torch.cat(r_t_collection, 1)

        tmp_s_t_target_collection, tmp_i_t_target_collection, tmp_r_t_target_collection = [], [], []
        for ii in range(k.n):
            tmp_s_t_target = torch.zeros([self.config.T_N, 1]).to(self.config.device)
            tmp_i_t_target = torch.zeros([self.config.T_N, 1]).to(self.config.device)
            tmp_r_t_target = torch.zeros([self.config.T_N, 1]).to(self.config.device)
            for jj in range(k.n):
                tmp_s_t_target -= (k.beta * s[:, ii:ii + 1] * k.M[ii][jj] * i[:, jj:jj + 1]) / k.N
                tmp_i_t_target += (k.beta * s[:, ii:ii + 1] * k.M[ii][jj] * i[:, jj:jj + 1]) / k.N
            tmp_i_t_target -= k.gamma * i[:, ii:ii + 1]
            tmp_r_t_target += k.gamma * i[:, ii:ii + 1]
            # tmp_s_t_target += (- self.config.mu * s[:, ii:ii+1] + self.config.lam)
            # tmp_i_t_target += (- self.config.mu * i[:, ii:ii+1])
            # tmp_r_t_target += (- self.config.mu * r[:, ii:ii+1])
            tmp_s_t_target_collection.append(tmp_s_t_target)
            tmp_i_t_target_collection.append(tmp_i_t_target)
            tmp_r_t_target_collection.append(tmp_r_t_target)
        s_t_target = torch.cat(tmp_s_t_target_collection, 1)
        i_t_target = torch.cat(tmp_i_t_target_collection, 1)
        r_t_target = torch.cat(tmp_r_t_target_collection, 1)

        f_s = - s_t + s_t_target
        f_i = - i_t + i_t_target
        f_r = - r_t + r_t_target

        return torch.cat((f_s, f_i, f_r), 1)


    def loss(self, y):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 10.0 * (self.criterion(ode_n, zeros_nD))

        loss3 = (1.0 if self.config.boundary else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]), y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]), self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))
        #(self.criterion(torch.abs(y[:, :, 0] - 0), y[:, :, 0] - 0) + self.criterion(
            # torch.abs(0.65 - y[:, :, 0]), 0.65 - y[:, :, 0]) + self.criterion(torch.abs(y[:, :, 1] - 1.2),
            #                                                                   y[:, :, 1] - 1.2) + self.criterion(
            # torch.abs(4.0 - y[:, :, 1]), 4.0 - y[:, :, 1]))
        # loss4 = (1.0 if self.config.penalty else 0.0) * sum([penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])
        # y_norm = torch.zeros(self.config.prob_dim).to(self.config.device)
        # for i in range(self.config.prob_dim):
        #     y_norm[i] = torch.var(
        #         (y[0, :, i] - torch.min(y[0, :, i])) / (torch.max(y[0, :, i]) - torch.min(y[0, :, i])))
        # loss4 = (1.0 if self.config.penalty else 0) * torch.mean(penalty_func(y_norm))
        # loss4 = self.criterion(1 / u_0, pt_all_zeros_3)
        # loss5 = self.criterion(torch.abs(u_0 - v_0), u_0 - v_0)

        loss = loss1 + loss2 + loss3
        loss_list = [loss1, loss2, loss3]
        return loss, loss_list

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        self.train()

        start_time = time.time()
        start_time_0 = start_time
        loss_record = []
        real_loss_mse_record = []
        real_loss_nmse_record = []
        time_record = []

        for epoch in range(1, self.config.args.iteration + 1):
            optimizer.zero_grad()

            y = self.forward(self.config.x)
            loss, loss_list = self.loss(y)
            loss_record.append(loss.item())
            real_loss_mse, real_loss_nmse = self.real_loss(y)
            real_loss_mse_record.append(real_loss_mse.item())
            real_loss_nmse_record.append(real_loss_nmse.item())

            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)  # retain_graph=True
            optimizer.step()
            scheduler.step()

            now_time = time.time()
            time_record.append(now_time - start_time_0)

            if epoch % self.config.args.epoch_step == 0:
                loss_print_part = " ".join(
                    ["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
                myprint(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
                        epoch, self.config.args.iteration, loss.item(), loss_print_part,
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                        (now_time - start_time_0) / 60.0 / epoch * (self.config.args.iteration - epoch)), self.config.args.log_path)
                start_time = now_time

                if epoch % self.config.args.test_step == 0:
                    self.y_tmp = y
                    self.epoch_tmp = epoch
                    self.loss_record_tmp = loss_record
                    self.real_loss_mse_record_tmp = real_loss_mse_record
                    self.real_loss_nmse_record_tmp = real_loss_nmse_record
                    self.time_record_tmp = time_record
                    self.test_model()
                    # save_path_loss = "{}/{}_{}_loss.npy".format(self.train_save_path_folder, self.config.model_name, self.time_string)
                    # np.save(save_path_loss, np.asarray(loss_record))

                    myprint("saving training info ...", self.config.args.log_path)
                    train_info = {
                        "model_name": self.config.model_name,
                        "seed": self.config.seed,
                        "prob_dim": self.config.prob_dim,
                        "activation": self.config.activation,
                        "cyclic": self.config.cyclic,
                        "stable": self.config.stable,
                        "derivative": self.config.derivative,
                        "loss_average_length": self.config.loss_average_length,
                        "epoch": self.config.args.iteration,
                        "epoch_stop": self.epoch_tmp,
                        "initial_lr": self.config.args.initial_lr,
                        "loss_length": len(loss_record),
                        "loss": np.asarray(loss_record),
                        "real_loss_mse": np.asarray(real_loss_mse_record),
                        "real_loss_nmse": np.asarray(real_loss_nmse_record),
                        "time": np.asarray(time_record),
                        "y_predict": y[0, :, :].cpu().detach().numpy(),
                        "y_truth": np.asarray(self.config.truth),
                        "y_shape": self.config.truth.shape,
                        # "config": self.config,
                        "time_string": self.time_string,
                        "weights_raw": np.asarray([
                            self.activate_block1.activate_weights_raw.cpu().detach().numpy(),
                            self.activate_block2.activate_weights_raw.cpu().detach().numpy(),
                            self.activate_block3.activate_weights_raw.cpu().detach().numpy(),
                            self.activate_block4.activate_weights_raw.cpu().detach().numpy(),
                        ]),
                        "weights": np.asarray([
                            self.activate_block1.activate_weights.cpu().detach().numpy(),
                            self.activate_block2.activate_weights.cpu().detach().numpy(),
                            self.activate_block3.activate_weights.cpu().detach().numpy(),
                            self.activate_block4.activate_weights.cpu().detach().numpy(),
                        ]),
                        "sin_weight": np.asarray([
                            self.activate_block1.activates[0].omega.cpu().detach().numpy(),
                            self.activate_block2.activates[0].omega.cpu().detach().numpy(),
                            self.activate_block3.activates[0].omega.cpu().detach().numpy(),
                            self.activate_block4.activates[0].omega.cpu().detach().numpy(),
                        ]),
                    }
                    train_info_path_loss = "{}/{}_{}_info.npy".format(self.train_save_path_folder,
                                                                      self.config.model_name, self.time_string)
                    model_save_path = "{}/{}_{}_last.pt".format(self.train_save_path_folder,
                                                                      self.config.model_name, self.time_string)
                    with open(train_info_path_loss, "wb") as f:
                        pickle.dump(train_info, f)
                    torch.save({
                        "model_state_dict": self.state_dict(),
                        "info": train_info,
                    }, model_save_path)

                    if epoch == self.config.args.iteration or self.early_stop():
                        myprint(str(train_info), self.config.args.log_path)
                        self.write_finish_log()
                        break

                    myprint(str(train_info), self.config.args.log_path)

    def test_model(self):
        if self.config.skip_draw_flag:
            myprint("(Skipped drawing)", self.config.args.log_path)
            return

        y_draw = self.y_tmp[0].cpu().detach().numpy().swapaxes(0, 1)
        x_draw = self.config.t
        y_draw_truth = self.config.truth.swapaxes(0, 1)
        save_path = "{}/{}_{}_epoch={}.png".format(self.figure_save_path_folder, self.config.model_name,
                                                   self.time_string, self.epoch_tmp)
        draw_two_dimension(
            y_lists=np.concatenate([y_draw, y_draw_truth], axis=0),
            x_list=x_draw,
            color_list=self.default_colors[: 2 * self.config.prob_dim],
            legend_list=self.config.curve_names + ["{}_true".format(item) for item in self.config.curve_names],
            line_style_list=["solid"] * self.config.prob_dim + ["dashed"] * self.config.prob_dim,
            fig_title="{}_{}_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp),
            fig_size=(8, 6),
            show_flag=False,
            save_flag=True,
            save_path=save_path,
            save_dpi=300,
            legend_loc="center right",
        )
        myprint("Figure is saved to {}".format(save_path), self.config.args.log_path)
        # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])

    def write_finish_log(self):
        with open("saves/record_omega.txt", "a") as f:
            f.write("{0},{1},{2},{3:.2f},{4},{5:.12f},{6:.12f},{7:.12f},{8},{9},{10},{11},{12},{13},{14}\n".format(
                self.config.model_name,  # 0
                self.time_string,  # 1
                self.config.seed,  # 2
                self.time_record_tmp[-1] / 60.0,  # 3
                self.config.args.iteration,  # 4
                sum(self.loss_record_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 5
                sum(self.real_loss_mse_record_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 6
                sum(self.real_loss_nmse_record_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 7
                self.config.pinn,  # 8
                self.config.activation,  # 9
                self.config.stable,  # 10
                self.config.cyclic,  # 11
                self.config.derivative,  # 12
                self.config.loss_average_length,  # 13
                "{}-{}".format(self.config.args.iteration - self.config.loss_average_length, self.config.args.iteration),  # 14
            ))

    @staticmethod
    def draw_loss_multi(loss_list, last_rate_list):
        m = MultiSubplotDraw(row=1, col=len(last_rate_list), fig_size=(8 * len(last_rate_list), 6),
                             tight_layout_flag=True, show_flag=False, save_flag=False, save_path=None)
        for one_rate in last_rate_list:
            m.add_subplot(
                y_lists=[loss_list[-int(len(loss_list) * one_rate):]],
                x_list=range(len(loss_list) - int(len(loss_list) * one_rate) + 1, len(loss_list) + 1),
                color_list=["blue"],
                line_style_list=["solid"],
                fig_title="Loss - lastest ${}$% - epoch ${}$ to ${}$".format(int(100 * one_rate), len(loss_list) - int(
                    len(loss_list) * one_rate) + 1, len(loss_list)),
                fig_x_label="epoch",
                fig_y_label="loss",
            )
        m.draw()


class PINNModel(FourierModel):
    def __init__(self, config):
        config.model_name = config.model_name.replace("Fourier", "PINN")
        super(PINNModel, self).__init__(config)

        self.fc1 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc5 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc6 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc7 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc8 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc9 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc10 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc11 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc12 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc13 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc14 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc15 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )


    def forward(self, x):
        x1_new = self.fc1(x)
        x2_new = self.fc2(x)
        x3_new = self.fc3(x)
        x4_new = self.fc4(x)
        x5_new = self.fc5(x)
        x6_new = self.fc6(x)
        x7_new = self.fc7(x)
        x8_new = self.fc8(x)
        x9_new = self.fc9(x)
        x10_new = self.fc10(x)
        x11_new = self.fc11(x)
        x12_new = self.fc12(x)
        x13_new = self.fc13(x)
        x14_new = self.fc14(x)
        x15_new = self.fc15(x)
        x = torch.cat((x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new, x9_new, x10_new, x11_new, x12_new, x13_new, x14_new, x15_new), -1)
        return x


def run(args, model_class):
    config = Config()
    config.seed = args.seed
    config.activation = args.activation
    config.cyclic = args.cyclic
    config.stable = args.stable
    config.derivative = args.derivative
    config.skip_draw_flag = args.skip_draw_flag
    config.pinn = args.pinn
    config.args.main_path = args.main_path
    config.args.log_path = args.log_path
    model = model_class(config).to(config.device)
    model.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    parser.add_argument("--main_path", default="./", help="main_path")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--pinn", type=int, default=0, help="0=off 1=on")
    parser.add_argument("--activation", choices=["gelu", "elu", "relu", "sin", "tanh", "softplus", "adaptive"], type=str, help="activation plan")
    parser.add_argument("--cyclic", type=int, choices=[0, 1], help="0=off 1=on")
    parser.add_argument("--stable", type=int, choices=[0, 1], help="0=off 1=on")
    parser.add_argument("--derivative", type=int, choices=[0, 1], help="0=off 1=on")
    parser.add_argument("--skip_draw_flag", type=int, default=1, choices=[0, 1], help="0=off 1=on")
    # parser.add_argument("--strategy", type=int, default=0, help="0=ones 1=fixed 2=adaptive")
    # parser.add_argument("--layer", type=int, default=8, help="number of layer")
    opt = parser.parse_args()
    opt.overall_start = get_now_string()

    myprint("log_path: {}".format(opt.log_path), opt.log_path)
    myprint("cuda is available: {}".format(torch.cuda.is_available()), opt.log_path)

    # if not opt.pinn:
    #     run(opt, FourierModel)
    # else:
    #     run(opt, PINNModel)
    try:
        if not opt.pinn:
            run(opt, FourierModel)
        else:
            run(opt, PINNModel)
    except Exception as e:
        print("[Error]", e)
