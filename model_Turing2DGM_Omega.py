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
import torchdiffeq

# from utils import *
import matplotlib.pyplot as plt
from template_omega import *


class Parameters:
    # N = 100
    # M = 1
    # d1 = 1
    # d2 = 40
    # c1 = 0.1  # 0.1
    # c2 = 0.9  # 0.9
    # c_1 = 1
    # c3 = 1
    # l = 0.6
    # w = 0.6

    N = 30
    M = 30
    l = 1.0
    w = 1.0
    d1 = 0.1
    d2 = 2
    c1 = 0.0
    c2 = 0.5
    c3 = 0.5
    c4 = 0.5
    c5 = 0.45
    k = 0.238


class TrainArgs:
    iteration = 5000  # 3000->10000
    epoch_step = 50  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.001
    ignore_save_flag = True
    main_path = "."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config(ConfigTemplate):
    def __init__(self):
        super(Config, self).__init__()
        self.model_name = "Turing2DGM_Fourier_Omega"
        self.curve_names = ["U", "V"]
        self.params = Parameters
        self.args = TrainArgs

        self.T_before = 30
        self.T = 100
        self.T_unit = 2e-2
        self.T_N_before = int(self.T_before / self.T_unit)
        self.T_N = int(self.T / self.T_unit)
        # self.y0 = np.asarray([50.0, 40.0, 10.0])
        self.boundary_list = np.asarray([[0.01, 2.3], [0.15, 2.2]])  # [0.1, 3.5], [0.3, 1.5]
        self.noise_rate = 0.05

        self.truth_torch = None

        self.setup()

        self.modes1 = 12  # 8
        self.modes2 = 12
        self.modes3 = 12
        if self.params.N < self.modes2:
            self.modes2 = self.params.N
        if self.params.M < self.modes3:
            self.modes3 = self.params.M
        self.width = 32  # 20

    def setup(self):
        self.setup_seed(0)
        self.prob_dim = len(self.curve_names)
        self.y0_before = torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device)
        self.y0_before[:, :, 0] += 1.0
        self.y0_before[:, :, 1] += 0.5
        self.t_before = np.asarray([i * self.T_unit for i in range(self.T_N_before)])
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        x = torch.zeros([1, self.T_N, self.params.N, self.params.M, 1]).to(self.device)
        self.x = FNO3d.get_grid(x.shape, x.device)

        truth_before = torchdiffeq.odeint(self.pend, self.y0_before.cpu(), torch.tensor(self.t_before),
                                          method='euler').to(self.device)
        noise = (torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) - 0.5) * self.noise_rate
        self.y0 = torch.abs(truth_before[-1] * (1.0 + noise) + 0.2)
        self.truth_torch = torchdiffeq.odeint(self.pend, self.y0.cpu(), torch.tensor(self.t), method='euler').to(
            self.device)
        # np.save(truth_path, self.truth.cpu().detach().numpy())

        print("y0:")
        # self.draw_turing(self.y0)
        print("Truth:")
        print("Truth U: max={0:.6f} min={1:.6f}".format(torch.max(self.truth_torch[:, :, :, 0]).item(),
                                                        torch.min(self.truth_torch[:, :, :, 0]).item()))
        print("Truth V: max={0:.6f} min={1:.6f}".format(torch.max(self.truth_torch[:, :, :, 1]).item(),
                                                        torch.min(self.truth_torch[:, :, :, 1]).item()))

        self.truth = self.truth_torch.cpu().detach().numpy()
        # self.draw_turing(self.truth_torch[-1])
        # turing_1d_all = self.truth_torch.reshape([-1, self.params.N, 2])
        # self.draw_turing_1d(turing_1d_all)
        self.loss_average_length = int(0.1 * self.args.iteration)

    def pend(self, t, y):
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], 2])

        reaction_part[:, :, 0] = self.params.c1 - self.params.c2 * y[:, :, 0] + self.params.c3 * (y[:, :, 0] ** 2) / ((1 + self.params.k * y[:, :, 0] ** 2) * y[:, :, 1])
        reaction_part[:, :, 1] = self.params.c4 * y[:, :, 0] ** 2 - self.params.c5 * y[:, :, 1]

        y_from_left = torch.roll(y, 1, 1)
        y_from_left[:, :1] = y[:, :1]
        y_from_right = torch.roll(y, -1, 1)
        y_from_right[:, -1:] = y[:, -1:]

        y_from_top = torch.roll(y, 1, 0)
        y_from_top[:1, :] = y[:1, :]
        y_from_bottom = torch.roll(y, -1, 0)
        y_from_bottom[-1:, :] = y[-1:, :]

        diffusion_part = torch.zeros([shapes[0], shapes[1], 2])
        diffusion_part[:, :, 0] = self.params.d1 * (
                ((y_from_left[:, :, 0] + y_from_right[:, :, 0] - y[:, :, 0] * 2) / (self.params.l ** 2)) + (
                (y_from_top[:, :, 0] + y_from_bottom[:, :, 0] - y[:, :, 0] * 2) / (self.params.w ** 2)))
        diffusion_part[:, :, 1] = self.params.d2 * (
                ((y_from_left[:, :, 1] + y_from_right[:, :, 1] - y[:, :, 1] * 2) / (self.params.l ** 2)) + (
                (y_from_top[:, :, 1] + y_from_bottom[:, :, 1] - y[:, :, 1] * 2) / (self.params.w ** 2)))
        return reaction_part + diffusion_part

    @staticmethod
    def draw_turing(map):
        # map: N * M * 2
        u = map[:, :, 0].cpu().detach().numpy()
        v = map[:, :, 1].cpu().detach().numpy()
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(u, cmap=plt.cm.jet, aspect='auto')
        ax1.set_title("u")
        cb1 = plt.colorbar(im1, shrink=1)

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(v, cmap=plt.cm.jet, aspect='auto')
        ax2.set_title("v")
        cb2 = plt.colorbar(im2, shrink=1)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_turing_1d(map):
        # map: N * M * 2
        u = map[:, :, 0].cpu().detach().numpy()
        v = map[:, :, 1].cpu().detach().numpy()
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(u, cmap=plt.cm.jet, aspect='auto', vmin=np.min(u[-1]), vmax=np.max(u[-1]))
        ax1.set_title("u")
        cb1 = plt.colorbar(im1, shrink=1)

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(v, cmap=plt.cm.jet, aspect='auto', vmin=np.min(v[-1]), vmax=np.max(v[-1]))
        ax2.set_title("v")
        cb2 = plt.colorbar(im2, shrink=1)
        plt.tight_layout()
        plt.show()


class FourierModel(FourierModelTemplate):
    def __init__(self, config):
        super(FourierModel, self).__init__(config)
        self.f_model = FNO3d(config)
        self.truth_loss()

    def truth_loss(self):
        y_truth = self.config.truth_torch.reshape(
            [1, self.config.T_N, self.config.params.N, self.config.params.M, self.config.prob_dim])
        y_truth = y_truth.to(self.config.device)
        # print("y_truth max:", torch.max(y_truth))
        # print("y_truth min:", torch.min(y_truth))
        tl, tl_list = self.loss(y_truth)
        loss_print_part = " ".join(
            ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(tl_list)])
        print("Ground truth has loss: Loss:{0:.12f} {1}".format(tl.item(), loss_print_part))

    def forward(self, x):
        return self.f_model(self.config.x)

    def real_loss(self, y):
        truth = self.config.truth_torch[:, :].to(self.config.device)
        real_loss_mse = self.criterion(y[0, :, :], truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y[0, :, :], truth) / (truth ** 2))
        return real_loss_mse, real_loss_nmse

    def ode_gradient(self, x, y):
        # y: 1 * T_N * N * M * 2
        y = y[0]
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], shapes[2], 2]).to(self.config.device)
        """
        reaction_part[:, :, 0] = self.params.c1 - self.params.c2 * y[:, :, 0] + self.params.c3 * (y[:, :, 0] ** 2) / ((1 + self.params.k * y[:, :, 0] ** 2) * y[:, :, 1])
        reaction_part[:, :, 1] = self.params.c4 * y[:, :, 0] ** 2 - self.params.c5 * y[:, :, 1]"""
        reaction_part[:, :, :, 0] = self.config.params.c1 - self.config.params.c2 * y[:, :, :,
                                                                                    0] + self.config.params.c3 * (
                                            y[:, :, :, 0] ** 2) / (
                                                (1 + self.config.params.k * y[:, :, :, 0] ** 2) * y[:, :, :, 1])
        reaction_part[:, :, :, 1] = self.config.params.c4 * y[:, :, :, 0] ** 2 - self.config.params.c5 * y[:, :, :, 1]

        y_from_left = torch.roll(y, 1, 2)
        y_from_left[:, :, :1] = y[:, :, :1]
        y_from_right = torch.roll(y, -1, 2)
        y_from_right[:, :, -1:] = y[:, :, -1:]

        y_from_top = torch.roll(y, 1, 1)
        y_from_top[:, :1, :] = y[:, :1, :]
        y_from_bottom = torch.roll(y, -1, 1)
        y_from_bottom[:, -1:, :] = y[:, -1:, :]

        diffusion_part = torch.zeros([shapes[0], shapes[1], shapes[2], 2]).to(self.config.device)
        diffusion_part[:, :, :, 0] = self.config.params.d1 * (((y_from_left[:, :, :, 0] + y_from_right[:, :, :, 0] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     0] * 2) / (
                                                                       self.config.params.l ** 2)) + ((y_from_top[:,
                                                                                                       :, :,
                                                                                                       0] + y_from_bottom[
                                                                                                            :, :, :,
                                                                                                            0] - y[
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 0] * 2) / (
                                                                                                              self.config.params.w ** 2)))
        diffusion_part[:, :, :, 1] = self.config.params.d2 * (((y_from_left[:, :, :, 1] + y_from_right[:, :, :, 1] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     1] * 2) / (
                                                                       self.config.params.l ** 2)) + ((y_from_top[:,
                                                                                                       :, :,
                                                                                                       1] + y_from_bottom[
                                                                                                            :, :, :,
                                                                                                            1] - y[
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 1] * 2) / (
                                                                                                              self.config.params.w ** 2)))

        y_t_theory = reaction_part + diffusion_part

        y_t = torch.gradient(y, spacing=(self.config.t_torch,), dim=0)[0]

        return y_t - y_t_theory

    def loss(self, y, iteration=-1):
        y0_pred = y[0, 0]
        y0_true = self.config.y0

        ode_y = self.ode_gradient(None, y)
        zeros_nD = torch.zeros([self.config.T_N, self.config.params.N, self.config.params.M, self.config.prob_dim]).to(
            self.config.device)

        loss1 = 1 * self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * self.criterion(ode_y, zeros_nD)

        # loss3 = 1 * (self.criterion(torch.abs(y - 0.1), y - 0.1) + self.criterion(torch.abs(6.5 - y), 6.5 - y))
        # loss4 = self.criterion(1e-3 / (y[0, :, :] ** 2 + 1e-10), zeros_nD)
        # self.criterion(1e-3 / (ode_1 ** 2 + 1e-10), zeros_1D) + self.criterion(1e-3 / (ode_2 ** 2 + 1e-10), zeros_1D) + self.criterion(1e-3 / (ode_3 ** 2 + 1e-10), zeros_1D)
        # loss5 = self.criterion(torch.abs(u_0 - v_0), u_0 - v_0)

        boundary_iteration = int(
            0.3 * self.config.args.iteration)  # 1.0 if self.config.boundary and iteration > boundary_iteration else 0.0
        loss3 = (1.0 if self.config.boundary and iteration > boundary_iteration else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]),
                           y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]),
                           self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))

        loss = loss1 + loss2 + loss3
        loss_list = [loss1, loss2, loss3]
        return loss, loss_list

    # def train_model(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr, weight_decay=0)
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
    #     self.train()
    #
    #     start_time = time.time()
    #     start_time_0 = start_time
    #     loss_record = []
    #     real_loss_mse_record = []
    #     real_loss_nmse_record = []
    #     time_record = []
    #
    #     for epoch in range(1, self.config.args.iteration + 1):
    #         optimizer.zero_grad()
    #
    #         y = self.f_model(self.config.x)
    #         loss, loss_list = self.loss(y)
    #         loss_record.append(loss.item())
    #         real_loss_mse, real_loss_nmse = self.real_loss(y)
    #         real_loss_mse_record.append(real_loss_mse.item())
    #         real_loss_nmse_record.append(real_loss_nmse.item())
    #
    #         torch.autograd.set_detect_anomaly(True)
    #         loss.backward(retain_graph=True)  # retain_graph=True
    #         optimizer.step()
    #         scheduler.step()
    #
    #         now_time = time.time()
    #         time_record.append(now_time - start_time_0)
    #
    #         if epoch % self.config.args.epoch_step == 0:
    #             loss_print_part = " ".join(
    #                 ["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
    #             myprint(
    #                 "Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} {3} Lr:{4:.6f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
    #                     epoch, self.config.args.iteration, loss.item(), loss_print_part,
    #                     optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
    #                                                      (now_time - start_time_0) / 60.0 / epoch * (
    #                                                                  self.config.args.iteration - epoch)),
    #                 self.config.args.log_path)
    #             start_time = now_time
    #
    #             if epoch % self.config.args.test_step == 0:
    #                 self.y_tmp = y
    #                 self.epoch_tmp = epoch
    #                 self.loss_record_tmp = loss_record
    #                 self.real_loss_mse_record_tmp = real_loss_mse_record
    #                 self.real_loss_nmse_record_tmp = real_loss_nmse_record
    #                 self.time_record_tmp = time_record
    #                 self.test_model()
    #                 # save_path_loss = "{}/{}_{}_loss.npy".format(self.train_save_path_folder, self.config.model_name, self.time_string)
    #                 # np.save(save_path_loss, np.asarray(loss_record))
    #
    #                 myprint("saving training info ...", self.config.args.log_path)
    #                 train_info = {
    #                     "model_name": self.config.model_name,
    #                     "seed": self.config.seed,
    #                     "layer": self.config.layer,
    #                     "activation": self.config.activation,
    #                     "penalty": self.config.penalty,
    #                     "strategy": self.config.strategy,
    #                     "epoch": self.config.args.iteration,
    #                     "epoch_stop": self.epoch_tmp,
    #                     "loss_length": len(loss_record),
    #                     "loss": np.asarray(loss_record),
    #                     "real_loss_mse": np.asarray(real_loss_mse_record),
    #                     "real_loss_nmse": np.asarray(real_loss_nmse_record),
    #                     "time": np.asarray(time_record),
    #                     "y_predict": y[0, :, :].cpu().detach().numpy(),
    #                     "y_truth": np.asarray(self.config.truth.cpu().detach().numpy()),
    #                     "y_shape": self.config.truth.cpu().detach().numpy().shape,
    #                     # "config": self.config,
    #                     "time_string": self.time_string,
    #                     "initial_lr": self.config.args.initial_lr,
    #                     # "weights_raw": np.asarray([
    #                     #     self.activate_block1.activate_weights_raw.cpu().detach().numpy(),
    #                     #     self.activate_block2.activate_weights_raw.cpu().detach().numpy(),
    #                     #     self.activate_block3.activate_weights_raw.cpu().detach().numpy(),
    #                     #     self.activate_block4.activate_weights_raw.cpu().detach().numpy(),
    #                     # ]),
    #                     # "weights": np.asarray([
    #                     #     self.activate_block1.activate_weights.cpu().detach().numpy(),
    #                     #     self.activate_block2.activate_weights.cpu().detach().numpy(),
    #                     #     self.activate_block3.activate_weights.cpu().detach().numpy(),
    #                     #     self.activate_block4.activate_weights.cpu().detach().numpy(),
    #                     # ]),
    #                     # "balance_weights": np.asarray([
    #                     #     self.activate_block1.balance_weights.cpu().detach().numpy(),
    #                     #     self.activate_block2.balance_weights.cpu().detach().numpy(),
    #                     #     self.activate_block3.balance_weights.cpu().detach().numpy(),
    #                     #     self.activate_block4.balance_weights.cpu().detach().numpy(),
    #                     # ]),
    #                     # "sin_weight": np.asarray([
    #                     #     self.activate_block1.activates[0].omega.cpu().detach().numpy(),
    #                     #     self.activate_block2.activates[0].omega.cpu().detach().numpy(),
    #                     #     self.activate_block3.activates[0].omega.cpu().detach().numpy(),
    #                     #     self.activate_block4.activates[0].omega.cpu().detach().numpy(),
    #                     # ]),
    #                 }
    #                 train_info_path_loss = "{}/{}_{}_info.npy".format(self.train_save_path_folder,
    #                                                                   self.config.model_name, self.time_string)
    #                 model_save_path = "{}/{}_{}_last.pt".format(self.train_save_path_folder,
    #                                                             self.config.model_name, self.time_string)
    #                 with open(train_info_path_loss, "wb") as f:
    #                     pickle.dump(train_info, f)
    #                 torch.save({
    #                     "model_state_dict": self.state_dict(),
    #                     "info": train_info,
    #                 }, model_save_path)
    #
    #                 if epoch == self.config.args.iteration or self.early_stop():
    #                     myprint(str(train_info), self.config.args.log_path)
    #                     self.write_finish_log()
    #                     break
    #
    #                 # myprint(str(train_info), self.config.args.log_path)
    #
    # def test_model(self):
    #
    #     u_draw_all = self.y_tmp[0, :, :, :, 0].reshape(self.config.T_N,
    #                                                    self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
    #         0, 1)[[10 * i for i in range(10)]]
    #     u_draw_all_truth = self.config.truth[:, :, :, 0].reshape(self.config.T_N,
    #                                                              self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
    #         0, 1)[[10 * i for i in range(10)]]
    #     v_draw_all = self.y_tmp[0, :, :, :, 1].reshape(self.config.T_N,
    #                                                    self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
    #         0, 1)[[10 * i for i in range(10)]]
    #     v_draw_all_truth = self.config.truth[:, :, :, 1].reshape(self.config.T_N,
    #                                                              self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
    #         0, 1)[[10 * i for i in range(10)]]
    #     x_draw = self.config.t
    #     draw_n = len(u_draw_all)
    #     save_path_2D = "{}/{}_{}_epoch={}_2D.png".format(self.figure_save_path_folder, self.config.model_name,
    #                                                      self.time_string, self.epoch_tmp)
    #
    #     m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 6), tight_layout_flag=True, show_flag=False, save_flag=True,
    #                          save_path=save_path_2D)
    #     m.add_subplot(
    #         y_lists=np.concatenate([u_draw_all, u_draw_all_truth], axis=0),
    #         x_list=x_draw,
    #         color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
    #         line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
    #         fig_title="{}_{}_U_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
    #         line_width=0.5)
    #     m.add_subplot(
    #         y_lists=np.concatenate([v_draw_all, v_draw_all_truth], axis=0),
    #         x_list=x_draw,
    #         color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
    #         line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
    #         fig_title="{}_{}_V_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
    #         line_width=0.5, )
    #     m.draw()
    #     u = self.y_tmp[0, :, :, :, 0].cpu().detach().numpy()
    #     v = self.y_tmp[0, :, :, :, 1].cpu().detach().numpy()
    #     u_last = u[-1]
    #     v_last = v[-1]
    #     u_true = self.config.truth[:, :, :, 0].cpu().detach().numpy()
    #     v_true = self.config.truth[:, :, :, 1].cpu().detach().numpy()
    #     u_last_true = u_true[-1]
    #     v_last_true = v_true[-1]
    #     save_path_map_all = "{}/{}_{}_epoch={}_map_all.png".format(self.figure_save_path_folder, self.config.model_name,
    #                                                                self.time_string, self.epoch_tmp)
    #     save_path_map_pred_only = "{}/{}_{}_epoch={}_map_pred_only.png".format(self.figure_save_path_folder,
    #                                                                            self.config.model_name, self.time_string,
    #                                                                            self.epoch_tmp)
    #     m = MultiSubplotDraw(row=2, col=2, fig_size=(16, 14), tight_layout_flag=True, save_flag=True,
    #                          save_path=save_path_map_all)
    #     m.add_subplot_turing(
    #         matrix=u_last,
    #         v_max=u_last_true.max(),
    #         v_min=u_last_true.min(),
    #         fig_title_size=10,
    #         number_label_size=10,
    #         fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
    #     m.add_subplot_turing(
    #         matrix=v_last,
    #         v_max=v_last_true.max(),
    #         v_min=v_last_true.min(),
    #         fig_title_size=10,
    #         number_label_size=10,
    #         fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
    #     m.add_subplot_turing(
    #         matrix=u_last_true,
    #         v_max=u_last_true.max(),
    #         v_min=u_last_true.min(),
    #         fig_title_size=10,
    #         number_label_size=10,
    #         fig_title="{}_{}_U_true".format(self.config.model_name, self.time_string))
    #     m.add_subplot_turing(
    #         matrix=v_last_true,
    #         v_max=v_last_true.max(),
    #         v_min=v_last_true.min(),
    #         fig_title_size=10,
    #         number_label_size=10,
    #         fig_title="{}_{}_V_true".format(self.config.model_name, self.time_string))
    #     m.draw()
    #
    #     m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 7), tight_layout_flag=True, show_flag=False, save_flag=True,
    #                          save_path=save_path_map_pred_only)
    #     m.add_subplot_turing(
    #         matrix=u_last,
    #         v_max=u_last_true.max(),
    #         v_min=u_last_true.min(),
    #         fig_title_size=10,
    #         number_label_size=10,
    #         fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
    #     m.add_subplot_turing(
    #         matrix=v_last,
    #         v_max=v_last_true.max(),
    #         v_min=v_last_true.min(),
    #         fig_title_size=10,
    #         number_label_size=10,
    #         fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
    #     m.draw()
    #
    #     # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])


def block_turing():
    return nn.Sequential(
        nn.Linear(3, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
    )


class PINNModel(FourierModel):
    def __init__(self, config):
        config.model_name = config.model_name.replace("Fourier", "PINN")
        super(PINNModel, self).__init__(config)

        self.sequences_u = nn.Sequential(*[block_turing() for _ in range(self.config.params.N * self.config.params.M)])
        self.sequences_v = nn.Sequential(*[block_turing() for _ in range(self.config.params.N * self.config.params.M)])

    def forward(self, x):
        shapes = x.shape
        # print(shapes)
        results_u = torch.zeros([shapes[0], shapes[1], shapes[2], shapes[3], 1]).to(self.config.device)
        results_v = torch.zeros([shapes[0], shapes[1], shapes[2], shapes[3], 1]).to(self.config.device)
        for n in range(self.config.params.N):
            for m in range(self.config.params.M):
                results_u[0, :, n, m, :] = self.sequences_u[n * self.config.params.M + m](x[0, :, n, m, :])
                results_v[0, :, n, m, :] = self.sequences_v[n * self.config.params.M + m](x[0, :, n, m, :])
        y = torch.cat((results_u, results_v), -1)
        return y




if __name__ == "__main__":
    run(Config, FourierModel, PINNModel)
# def run(args, model_class):
#     config = Config()
#     config.seed = args.seed
#     config.layer = args.layer
#     config.activation = args.activation
#     config.penalty = args.penalty
#     config.pinn = args.pinn
#     config.strategy = args.strategy
#     config.args.main_path = args.main_path
#     config.args.log_path = args.log_path
#     model = model_class(config).to(config.device)
#     model.train_model()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
#     parser.add_argument("--main_path", default="./", help="main_path")
#     parser.add_argument("--seed", type=int, default=0, help="seed")
#     parser.add_argument("--pinn", type=int, default=0, help="0=off 1=on")
#     parser.add_argument("--activation", default="plan3", type=str, help="activation plan")
#     parser.add_argument("--penalty", type=int, default=1, help="0=off 1=on")
#     parser.add_argument("--strategy", type=int, default=0, help="0=ones 1=fixed 2=adaptive")
#     parser.add_argument("--layer", type=int, default=8, help="number of layer")
#     opt = parser.parse_args()
#     opt.overall_start = get_now_string()
#
#     myprint("log_path: {}".format(opt.log_path), opt.log_path)
#     myprint("cuda is available: {}".format(torch.cuda.is_available()), opt.log_path)
#
#     if not opt.pinn:
#         run(opt, FourierModel)
#     else:
#         run(opt, PINNModel)
#
#     # try:
#     #     if not opt.pinn:
#     #         run(opt, FourierModel)
#     #     else:
#     #         run(opt, PINNModel)
#     # except Exception as e:
#     #     print("[Error]", e)
