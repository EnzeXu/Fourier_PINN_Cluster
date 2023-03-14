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
    alpha = 1.00
    beta = 3.00
    gamma = 0.30
    delta = 0.1


class TrainArgs:
    iteration = 100000  # 20000 -> 50000
    epoch_step = 1000  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.01
    main_path = "."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config(ConfigTemplate):
    def __init__(self):
        super(Config, self).__init__()
        self.model_name = "PP_Fourier_Omega"
        self.curve_names = ["U", "V"]
        self.params = Parameters
        self.args = TrainArgs

        self.T = 20
        self.T_unit = 1e-3
        self.y0 = np.asarray([64.73002741, 6.13106793])
        # self.boundary_list = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        self.boundary_list = np.asarray([[9, 100], [0, 100]])

        self.setup()

    def pend(self, y, t):
        k = self.params
        dydt = np.asarray([
            self.params.alpha * y[0] - self.params.gamma * y[0] * y[1],
            - self.params.beta * y[1] + self.params.delta * y[0] * y[1]
        ])
        return dydt


# def penalty_cyclic_func(x):
#     return 1 * (- torch.tanh((x - 0.05) * 200) + 1)

def penalty_func(x):
    return 1 * (- torch.tanh((x - 0.004) * 300) + 1)


class FourierModel(FourierModelTemplate):
    def __init__(self, config):
        super(FourierModel, self).__init__(config)
        self.truth_loss()

    def real_loss(self, y):
        truth = torch.tensor(self.config.truth[:, :]).to(self.config.device)
        real_loss_mse = self.criterion(y[0, :, :], truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y[0, :, :], truth) / (truth ** 2))
        return real_loss_mse, real_loss_nmse


    def ode_gradient(self, x, y):
        k = self.config.params

        u = y[0, :, 0]
        v = y[0, :, 1]

        u_t = torch.gradient(u, spacing=(self.config.t_torch,))[0]
        v_t = torch.gradient(v, spacing=(self.config.t_torch,))[0]

        f_u = u_t - (k.alpha - k.gamma * v) * u
        f_v = v_t - (k.delta * u - k.beta) * v

        return torch.cat((f_u.reshape([-1, 1]), f_v.reshape([-1, 1])), 1)


    def loss(self, y, iteration=-1):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

        boundary_iteration = int(0.0 * self.config.args.iteration)  # 1.0 if self.config.boundary and iteration > boundary_iteration else 0.0
        loss3 = (1.0 if self.config.boundary and iteration > boundary_iteration else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]),
                           y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]),
                           self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))

        # y_norm = torch.zeros(self.config.prob_dim).to(self.config.device)
        # for i in range(self.config.prob_dim):
        #     y_norm[i] = torch.var((y[0, :, i] - torch.min(y[0, :, i])) / (torch.max(y[0, :, i]) - torch.min(y[0, :, i])))
        # loss4 = (1.0 if self.config.cyclic else 0) * torch.mean(penalty_cyclic_func(y_norm))

        loss4 = (1.0 if self.config.cyclic else 0) * sum(
            [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])
        # loss4 = (1.0 if self.config.cyclic else 0) * sum(
        #     [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])

        loss = loss1 + loss2 + loss3 + loss4
        loss_list = [loss1, loss2, loss3, loss4]
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


    def forward(self, x):
        x1_new = self.fc1(x)
        x2_new = self.fc2(x)
        x = torch.cat((x1_new, x2_new), -1)
        return x


if __name__ == "__main__":
    run(Config, FourierModel, PINNModel)
