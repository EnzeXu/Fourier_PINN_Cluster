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
    beta = 10
    n = 3


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
        self.model_name = "REP3_Fourier_Omega"
        self.curve_names = ["p_cI", "p_lacI", "p_tetR"]
        self.params = Parameters
        self.args = TrainArgs

        self.T = 10
        self.T_unit = 1e-2
        self.y0 = np.asarray([1.75241881, 4.77323806, 1.04664267])
        # self.boundary_list = np.asarray([[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]])
        self.boundary_list = np.asarray([[0.64, 5.5], [0.64, 5.5], [0.64, 5.5]])

        self.setup()

    def pend(self, y, t):
        k = self.params
        dydt = np.asarray([
            k.beta / (1 + y[2] ** k.n) - y[0],
            k.beta / (1 + y[0] ** k.n) - y[1],
            k.beta / (1 + y[1] ** k.n) - y[2]
        ])
        return dydt

# def penalty_cyclic_func(x):
#     return 1 * (- torch.tanh((x - 0.05) * 150) + 1)

def penalty_func(x):
    return 1 * (- torch.tanh((x - 0.07) * 200) + 1)

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

        p_cl = y[0, :, 0]
        p_lacl = y[0, :, 1]
        p_tetR = y[0, :, 2]

        p_cl_t = torch.gradient(p_cl, spacing=(self.config.t_torch,))[0]
        p_lacl_t = torch.gradient(p_lacl, spacing=(self.config.t_torch,))[0]
        p_tetR_t = torch.gradient(p_tetR, spacing=(self.config.t_torch,))[0]

        f_p_cl = p_cl_t - (k.beta / (1 + p_tetR ** k.n) - p_cl)
        f_p_lacl = p_lacl_t - (k.beta / (1 + p_cl ** k.n) - p_lacl)
        f_p_tetR = p_tetR_t - (k.beta / (1 + p_lacl ** k.n) - p_tetR)

        return torch.cat((f_p_cl.reshape([-1, 1]), f_p_lacl.reshape([-1, 1]), f_p_tetR.reshape([-1, 1])), 1)

    def loss(self, y, iteration=-1):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

        loss3 = (1.0 if self.config.boundary else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]),
                           y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]),
                           self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))

        # y_norm = torch.zeros(self.config.prob_dim).to(self.config.device)
        # for i in range(self.config.prob_dim):
        #     y_norm[i] = torch.var(
        #         (y[0, :, i] - torch.min(y[0, :, i])) / (torch.max(y[0, :, i]) - torch.min(y[0, :, i])))
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

        self.fc3 = nn.Sequential(
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
        x = torch.cat((x1_new, x2_new, x3_new), -1)
        return x


if __name__ == "__main__":
    run(Config, FourierModel, PINNModel)
