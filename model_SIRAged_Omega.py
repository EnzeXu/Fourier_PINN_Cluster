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
from template_omega import *


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
    initial_lr = 0.001
    main_path = "."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01



class Config(ConfigTemplate):
    def __init__(self):
        super(Config, self).__init__()

        self.model_name = "SIRAged_Fourier_Omega"
        self.params = Parameters
        self.curve_names = ["S{}".format(i) for i in range(1, self.params.n + 1)] + ["I{}".format(i) for i in range(1, self.params.n + 1)] + ["R{}".format(i) for i in range(1, self.params.n + 1)]
        self.args = TrainArgs

        self.T = 100
        self.T_unit = 1e-2
        self.y0 = np.asarray([50.0] * self.params.n + [40.0] * self.params.n + [10.0] * self.params.n)
        self.boundary_list = np.asarray([[0.0, 100.0]] * 3 * self.params.n)

        self.setup()

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


class FourierModel(FourierModelTemplate):
    def __init__(self, config):
        super(FourierModel, self).__init__(config)

    def real_loss(self, y):
        truth = torch.tensor(self.config.truth[:, 10:15]).to(self.config.device)
        real_loss_mse = self.criterion(y[0, :, 10:15], truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y[0, :, 10:15], truth) / (truth ** 2))
        return real_loss_mse, real_loss_nmse

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

        ode_n, dy = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

        loss3 = (1.0 if self.config.boundary else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]),
                           y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]),
                           self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))
        loss4 = (1.0 if self.config.cyclic else 0) * sum(
            [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])

        stable_period = 0.9
        loss5 = (1.0 if self.config.stable else 0) * self.criterion(
            torch.abs(0.1 - torch.abs(dy[int(stable_period * self.config.T_N):, :])),
            0.1 - torch.abs(dy[int(stable_period * self.config.T_N):, :]))
        # print("dy max", torch.max(dy[int(0.9 * self.config.T_N):,:]))
        # print("dy min", torch.min(dy[int(0.9 * self.config.T_N):,:]))
        # print("dy avg", torch.mean(dy[int(0.9 * self.config.T_N):,:]))

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        loss_list = [loss1, loss2, loss3, loss4, loss5]
        return loss, loss_list


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


if __name__ == "__main__":
    run(Config, FourierModel, PINNModel)

