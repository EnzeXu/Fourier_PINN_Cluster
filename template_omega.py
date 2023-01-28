import torch
import os
import time
import random
import numpy as np
import pickle
import argparse
import torch.nn as nn

from scipy.integrate import odeint

from utils import get_now_string, ColorCandidate, myprint, MultiSubplotDraw, draw_two_dimension


class ConfigTemplate:
    def __init__(self):
        self.model_name = "REP6_Fourier_Omega"
        self.curve_names = ["m_lacI", "m_tetR", "m_cI", "p_cI", "p_lacI", "p_tetR"]
        self.params = None
        self.args = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0
        self.pinn = 0

        self.T = 20
        self.T_unit = 1e-2
        self.T_N = None

        self.prob_dim = None
        self.y0 = None
        self.boundary_list = None
        self.t = None
        self.t_torch = None
        self.x = None
        self.truth = None

        self.modes = 64  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.width = 16
        self.fc_map_dim = 128

        self.activation = ""
        self.cyclic = None
        self.stable = None
        self.boundary = None
        self.derivative = None
        self.skip_draw_flag = False
        self.loss_average_length = None

    def setup(self):
        self.T_N = int(self.T / self.T_unit)
        self.prob_dim = len(self.curve_names)
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        self.x = torch.tensor(np.asarray([[[i * self.T_unit] * 1 for i in range(self.T_N)]]),
                              dtype=torch.float32).to(self.device)
        self.truth = odeint(self.pend, self.y0, self.t)
        self.loss_average_length = int(0.1 * self.args.iteration)

    def pend(self, y, t):
        dydt = np.zeros([1])
        return dydt


class FourierModelTemplate(nn.Module):
    def __init__(self, config):
        super(FourierModelTemplate, self).__init__()
        self.time_string = get_now_string()
        self.config = config
        self.setup_seed(self.config.seed)

        self.fc0 = nn.Linear(2, self.config.width)  # input channel is 2: (a(x), x)
        # self.layers = Layers(config=self.config, n=self.config.layer).to(self.config.device)
        self.conv0 = SpectralConv1d(self.config).to(self.config.device)
        self.conv1 = SpectralConv1d(self.config).to(self.config.device)
        self.conv2 = SpectralConv1d(self.config).to(self.config.device)
        self.conv3 = SpectralConv1d(self.config).to(self.config.device)
        self.w0 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.w1 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.w2 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.w3 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        self.mlp0 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        self.mlp1 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        self.mlp2 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        self.mlp3 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        self.activate_block0 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block3 = ActivationBlock(self.config).to(self.config.device)

        self.fc1 = nn.Linear(self.config.width, self.config.fc_map_dim)
        self.fc2 = nn.Linear(self.config.fc_map_dim, self.config.prob_dim)

        self.criterion = torch.nn.MSELoss().to(self.config.device)  # "sum"
        self.criterion_non_reduce = torch.nn.MSELoss(reduction="none").to(self.config.device)

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

    def truth_loss(self):
        y_truth = torch.tensor(self.config.truth.reshape([1, self.config.T_N, self.config.prob_dim])).to(
            self.config.device)
        tl, tl_list = self.loss(y_truth)
        print("tl_list=", tl_list)
        loss_print_part = " ".join(
            ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(tl_list)])
        myprint("Ground truth has loss: Loss:{0:.12f} {1}".format(tl.item(), loss_print_part), self.config.args.log_path)

    #  MSE-loss of predicted value against truth
    def real_loss(self, y):
        truth = torch.tensor(self.config.truth[:, :]).to(self.config.device)
        real_loss_mse = self.criterion(y[0, :, :], truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y[0, :, :], truth) / (truth ** 2))
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
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activate_block0(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activate_block1(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activate_block2(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activate_block3(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def ode_gradient(self, x, y):
        k = self.config.params

        m_lacl = y[0, :, 0]
        m_tetR = y[0, :, 1]
        m_cl = y[0, :, 2]
        p_cl = y[0, :, 3]
        p_lacl = y[0, :, 4]
        p_tetR = y[0, :, 5]

        m_lacl_t = torch.gradient(m_lacl, spacing=(self.config.t_torch,))[0]
        m_tetR_t = torch.gradient(m_tetR, spacing=(self.config.t_torch,))[0]
        m_cl_t = torch.gradient(m_cl, spacing=(self.config.t_torch,))[0]
        p_cl_t = torch.gradient(p_cl, spacing=(self.config.t_torch,))[0]
        p_lacl_t = torch.gradient(p_lacl, spacing=(self.config.t_torch,))[0]
        p_tetR_t = torch.gradient(p_tetR, spacing=(self.config.t_torch,))[0]

        f_m_lacl = m_lacl_t - (k.beta * (k.rho + 1 / (1 + p_tetR ** k.n)) - m_lacl)
        f_m_tetR = m_tetR_t - (k.beta * (k.rho + 1 / (1 + p_cl ** k.n)) - m_tetR)
        f_m_cl = m_cl_t - (k.beta * (k.rho + 1 / (1 + p_lacl ** k.n)) - m_cl)
        f_p_cl = p_cl_t - (k.gamma * (m_lacl - p_cl))
        f_p_lacl = p_lacl_t - (k.gamma * (m_tetR - p_lacl))
        f_p_tetR = p_tetR_t - (k.gamma * (m_cl - p_tetR))

        return torch.cat((f_m_lacl.reshape([-1, 1]), f_m_tetR.reshape([-1, 1]), f_m_cl.reshape([-1, 1]),
                          f_p_cl.reshape([-1, 1]), f_p_lacl.reshape([-1, 1]), f_p_tetR.reshape([-1, 1])), 1)

    def loss(self, y):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.args.iteration)
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
                    ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
                myprint(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.12f} {3} Lr:{4:.12f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
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
                            self.activate_block0.activate_weights_raw.cpu().detach().numpy(),
                            self.activate_block1.activate_weights_raw.cpu().detach().numpy(),
                            self.activate_block2.activate_weights_raw.cpu().detach().numpy(),
                            self.activate_block3.activate_weights_raw.cpu().detach().numpy(),
                        ]),
                        "weights": np.asarray([
                            self.activate_block0.activate_weights.cpu().detach().numpy(),
                            self.activate_block1.activate_weights.cpu().detach().numpy(),
                            self.activate_block2.activate_weights.cpu().detach().numpy(),
                            self.activate_block3.activate_weights.cpu().detach().numpy(),
                        ]),
                        "sin_weight": np.asarray([
                            self.activate_block0.activates[0].omega.cpu().detach().numpy(),
                            self.activate_block1.activates[0].omega.cpu().detach().numpy(),
                            self.activate_block2.activates[0].omega.cpu().detach().numpy(),
                            self.activate_block3.activates[0].omega.cpu().detach().numpy(),
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

                    # myprint(str(train_info), self.config.args.log_path)

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
            show_flag=True,
            save_flag=True,
            save_path=save_path,
            save_dpi=300,
            legend_loc="center right",
        )
        myprint("Figure is saved to {}".format(save_path), self.config.args.log_path)
        self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])
        self.draw_loss_multi(self.real_loss_nmse_record_tmp, [1.0, 0.5, 0.25, 0.125])

    def write_finish_log(self):
        with open(os.path.join(self.config.args.main_path, "saves/record_omega.txt"), "a") as f:
            f.write("{0},{1},{2},{3:.2f},{4},{5:.12f},{6:.12f},{7:.12f},{8},{9},{10},{11},{12},{13},{14},{15}\n".format(
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
                self.config.boundary,  # 12
                self.config.loss_average_length,  # 14
                "{}-{}".format(self.config.args.iteration - self.config.loss_average_length, self.config.args.iteration),  # 15
            ))

    @staticmethod
    def draw_loss_multi(loss_list, last_rate_list):
        m = MultiSubplotDraw(row=1, col=len(last_rate_list), fig_size=(8 * len(last_rate_list), 6),
                             tight_layout_flag=True, show_flag=True, save_flag=False, save_path=None)
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


class SpectralConv1d(nn.Module):
    def __init__(self, config):
        super(SpectralConv1d, self).__init__()
        self.config = config
        self.in_channels = self.config.width
        self.out_channels = self.config.width
        self.scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat).to(
            self.config.device)
        out_ft[:, :, :self.config.modes] = self.compl_mul1d(x_ft[:, :, :self.config.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = torch.nn.functional.gelu(x)
        x = self.mlp2(x)
        return x

class MySin(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return torch.sin(self.omega * x)


class MySoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 1

    def forward(self, x):
        return nn.Softplus(beta=self.beta)(x)


def activation_func(activation):
    return nn.ModuleDict({
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "selu": nn.SELU(),
        "sin": MySin(),
        "tanh": nn.Tanh(),
        "softplus": MySoftplus(),
        "elu": nn.ELU(),
        "none": nn.Identity(),
    })[activation]


class ActivationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activate_list = ["sin", "tanh", "relu", "gelu", "softplus", "elu"]
        self.activate_list_3 = ["gelu", "softplus", "elu"]
        assert self.config.activation in self.activate_list + ["adaptive", "adaptive_3"]
        self.activates = nn.ModuleList([activation_func(item).to(config.device) for item in self.activate_list])
        self.activate_weights_raw = nn.Parameter(torch.rand(len(self.activate_list)).to(self.config.device), requires_grad=True)
        self.activates_3 = nn.ModuleList([activation_func(item).to(config.device) for item in self.activate_list_3])
        self.activate_weights_raw_3 = nn.Parameter(torch.rand(3).to(self.config.device), requires_grad=True)

        self.my_sin = activation_func("sin")
        self.my_softplus = activation_func("softplus")

        self.activate_weights = my_softmax(self.activate_weights_raw)
        self.activate_weights_3 = my_softmax(self.activate_weights_raw_3)
        # assert self.config.strategy in [0, 1, 2]
        # if self.config.strategy == 0:
        #     self.balance_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device)
        # elif self.config.strategy == 1:
        #     self.balance_weights = torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device)
        # else:
        #     self.balance_weights = nn.Parameter(torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device), requires_grad=True)
        # print("self.activate_weights device = {}".format(self.activate_weights.device))

    def forward(self, x):
        if self.config.activation == "gelu":
            return nn.functional.gelu(x)
        elif self.config.activation == "relu":
            return nn.functional.relu(x)
        elif self.config.activation == "tanh":
            return nn.functional.tanh(x)
        elif self.config.activation == "elu":
            return nn.functional.elu(x)
        elif self.config.activation == "sin":
            return self.my_sin(x)
        elif self.config.activation == "softplus":
            return self.my_softplus(x)
        elif self.config.activation == "selu":
            return nn.functional.selu(x)
        activation_res = 0.0
        if self.config.activation == "adaptive":
            self.activate_weights = my_softmax(self.activate_weights_raw)
            for i in range(len(self.activate_list)):
                tmp_sum = self.activate_weights[i] * self.activates[i](x)
                activation_res += tmp_sum
        else: # adaptive_3
            self.activate_weights_3 = my_softmax(self.activate_weights_raw_3)
            for i in range(len(self.activate_list_3)):
                tmp_sum = self.activate_weights_3[i] * self.activates_3[i](x)
                activation_res += tmp_sum
        return activation_res


def my_softmax(x):
    exponent_vector = torch.exp(x)
    sum_of_exponents = torch.sum(exponent_vector)
    softmax_vector = exponent_vector / sum_of_exponents
    return softmax_vector


def run(config, fourier_model, pinn_model):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    parser.add_argument("--main_path", default="./", help="main_path")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--pinn", type=int, default=0, help="0=off 1=on")
    parser.add_argument("--activation", choices=["gelu", "elu", "relu", "sin", "tanh", "softplus", "adaptive", "adaptive_3", "selu"],
                        type=str, help="activation plan")
    parser.add_argument("--cyclic", type=int, choices=[0, 1, 2], help="0=off 1=on")
    parser.add_argument("--stable", type=int, choices=[0, 1], help="0=off 1=on")
    parser.add_argument("--derivative", type=int, choices=[0, 1], help="0=off 1=on")
    parser.add_argument("--boundary", type=int, choices=[0, 1, 2], help="0=off 1=on")
    parser.add_argument("--skip_draw_flag", type=int, default=1, choices=[0, 1], help="0=off 1=on")
    parser.add_argument("--test", type=int, default=0, help="test will take epoch as 10")
    # parser.add_argument("--strategy", type=int, default=0, help="0=ones 1=fixed 2=adaptive")
    # parser.add_argument("--layer", type=int, default=8, help="number of layer")
    opt = parser.parse_args()
    opt.overall_start = get_now_string()

    myprint("log_path: {}".format(opt.log_path), opt.log_path)
    myprint("cuda is available: {}".format(torch.cuda.is_available()), opt.log_path)

    config = config()
    config.seed = opt.seed
    config.activation = opt.activation
    config.cyclic = opt.cyclic
    config.stable = opt.stable
    config.derivative = opt.derivative
    config.boundary = opt.boundary
    config.skip_draw_flag = opt.skip_draw_flag
    config.pinn = opt.pinn
    config.args.main_path = opt.main_path
    config.args.log_path = opt.log_path
    if opt.test:
        config.args.iteration = 10
        config.args.epoch_step = 1  # 1000  # 1000
        config.args.test_step = config.args.epoch_step * 10
    try:
        if not opt.pinn:
            model = fourier_model(config).to(config.device)
        else:
            model = pinn_model(config).to(config.device)
        model.train_model()
    except Exception as e:
        print("[Error]", e)


def penalty_func(x):
    return 1 * (- torch.tanh((x - 0.005) * 200) + 1)  # 1 * (- torch.tanh((x - 0.004) * 300) + 1)
