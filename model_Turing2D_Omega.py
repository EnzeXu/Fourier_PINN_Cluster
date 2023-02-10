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
    N = 30
    M = 30
    d1 = 1
    d2 = 40
    c1 = 0.1  # 0.1
    c2 = 0.9  # 0.9
    c_1 = 1
    c3 = 1
    l = 0.6
    w = 0.6


class TrainArgs:
    iteration = 5000  # 3000->10000
    epoch_step = 10  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.01
    ignore_save_flag = True
    main_path = "."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config(ConfigTemplate):
    def __init__(self):
        super(Config, self).__init__()
        self.model_name = "Turing2D_Fourier_Omega"
        self.curve_names = ["U", "V"]
        self.params = Parameters
        self.args = TrainArgs

        self.T_before = 32  # 30
        self.T = 2
        self.T_unit = 2e-3
        self.T_N_before = int(self.T_before / self.T_unit)
        self.T_N = int(self.T / self.T_unit)
        # self.y0 = np.asarray([50.0, 40.0, 10.0])
        self.boundary_list = np.asarray([[0.1, 6.0], [0.2, 1.5]])  # [0.1, 3.5], [0.3, 1.5]
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
        self.y0_before = torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) + 2.0
        self.t_before = np.asarray([i * self.T_unit for i in range(self.T_N_before)])
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        x = torch.zeros([1, self.T_N, self.params.N, self.params.M, 1]).to(self.device)
        self.x = FNO3d.get_grid(x.shape, x.device)

        truth_before = torchdiffeq.odeint(self.pend, self.y0_before.cpu(), torch.tensor(self.t_before),
                                          method='euler').to(self.device)
        noise = (torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) - 0.5) * self.noise_rate
        self.y0 = truth_before[
                      -1] + 0.1  # torch.abs(truth_before[-1] * (1.0 + noise) + 0.1) # torch.abs(truth_before[-1] * (1.0 + noise) + 0.2)
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
        # self.draw_turing(self.truth[-1])
        self.truth = self.truth_torch.cpu().detach().numpy()
        # self.draw_turing(self.truth_torch[-1])
        # turing_1d_all = self.truth_torch.reshape([-1, self.params.N, 2])
        # self.draw_turing_1d(turing_1d_all)
        self.loss_average_length = int(0.1 * self.args.iteration)

    def pend(self, t, y):
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], 2])
        reaction_part[:, :, 0] = self.params.c1 - self.params.c_1 * y[:, :, 0] + self.params.c3 * (y[:, :, 0] ** 2) * y[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      1]
        reaction_part[:, :, 1] = self.params.c2 - self.params.c3 * (y[:, :, 0] ** 2) * y[:, :, 1]

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
        y = y[0]
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], shapes[2], 2]).to(self.config.device)
        reaction_part[:, :, :, 0] = self.config.params.c1 - self.config.params.c_1 * y[:, :, :,
                                                                                     0] + self.config.params.c3 * (
                                            y[:, :, :, 0] ** 2) * y[:, :, :, 1]
        reaction_part[:, :, :, 1] = self.config.params.c2 - self.config.params.c3 * (y[:, :, :, 0] ** 2) * y[:, :, :, 1]

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

        return y_t - y_t_theory, y_t

    def loss(self, y, iteration=-1):
        y0_pred = y[0, 0]
        y0_true = self.config.y0

        ode_y, dy = self.ode_gradient(None, y)
        zeros_nD = torch.zeros([self.config.T_N, self.config.params.N, self.config.params.M, self.config.prob_dim]).to(
            self.config.device)

        loss1 = 1 * self.criterion(y0_pred, y0_true)
        loss2 = 1 * self.criterion(ode_y, zeros_nD)

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
        # print("dy max", torch.max(dy[int(0.9 * self.config.T_N):]))
        # print("dy min", torch.min(dy[int(0.9 * self.config.T_N):]))
        # print("dy avg", torch.mean(dy[int(0.9 * self.config.T_N):]))
        stable_period = 0.9
        stable_iteration = int(0.3 * self.config.args.iteration)
        loss5 = (1.0 if self.config.stable and iteration > stable_iteration else 0) * self.criterion(
            torch.abs(0.13 - torch.abs(dy[int(stable_period * self.config.T_N):])),
            0.13 - torch.abs(dy[int(stable_period * self.config.T_N):]))
        loss = loss1 + loss2 + loss3 + loss5
        loss_list = [loss1, loss2, loss3, loss5]
        return loss, loss_list

    def test_model(self):
        if self.config.skip_draw_flag:
            myprint("(Skipped drawing)", self.config.args.log_path)
            return
        u_draw_all = self.y_tmp[0, :, :, :, 0].reshape(self.config.T_N,
                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        u_draw_all_truth = self.config.truth_torch[:, :, :, 0].reshape(self.config.T_N,
                                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        v_draw_all = self.y_tmp[0, :, :, :, 1].reshape(self.config.T_N,
                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        v_draw_all_truth = self.config.truth_torch[:, :, :, 1].reshape(self.config.T_N,
                                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        x_draw = self.config.t
        draw_n = len(u_draw_all)
        save_path_2D = "{}/{}_{}_epoch={}_2D.png".format(self.figure_save_path_folder, self.config.model_name,
                                                         self.time_string, self.epoch_tmp)

        m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 6), tight_layout_flag=True, show_flag=True, save_flag=True,
                             save_path=save_path_2D)
        m.add_subplot(
            y_lists=np.concatenate([u_draw_all, u_draw_all_truth], axis=0),
            x_list=x_draw,
            color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
            line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
            fig_title="{}_{}_U_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
            line_width=0.5)
        m.add_subplot(
            y_lists=np.concatenate([v_draw_all, v_draw_all_truth], axis=0),
            x_list=x_draw,
            color_list=[self.default_colors[0]] * draw_n + [self.default_colors[1]] * draw_n,
            line_style_list=["solid"] * draw_n + ["dashed"] * draw_n,
            fig_title="{}_{}_V_epoch={}_2D".format(self.config.model_name, self.time_string, self.epoch_tmp),
            line_width=0.5, )
        m.draw()
        self.config.draw_turing_1d(self.y_tmp.reshape([-1, self.config.params.N, 2]))
        self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])
        self.draw_loss_multi(self.real_loss_nmse_record_tmp, [1.0, 0.5, 0.25, 0.125])


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
