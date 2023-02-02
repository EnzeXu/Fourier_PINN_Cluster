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
    beta = 0.01
    gamma = 0.05
    N = 100.0


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
        self.model_name = "SIR_Fourier_Omega"
        self.curve_names = ["S", "I", "R"]
        self.params = Parameters
        self.args = TrainArgs

        self.T = 100
        self.T_unit = 1e-2
        self.y0 = np.asarray([50.0, 40.0, 10.0])
        # self.boundary_list = np.asarray([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]])
        self.boundary_list = np.asarray([[0, 50.00], [0.63, 73.5], [10, 99.37]])

        self.setup()

    def pend(self, y, t):
        k = self.params
        dydt = np.asarray([
            - self.params.beta * y[0] * y[1],
            self.params.beta * y[0] * y[1] - self.params.gamma * y[1],
            self.params.gamma * y[1]
        ])
        return dydt


class FourierModel(FourierModelTemplate):
    def __init__(self, config):
        super(FourierModel, self).__init__(config)
        self.truth_loss()

    def real_loss(self, y):
        truth = torch.tensor(self.config.truth[:, 2]).to(self.config.device)
        real_loss_mse = self.criterion(y[0, :, 2], truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y[0, :, 2], truth) / (truth ** 2))
        return real_loss_mse, real_loss_nmse

    def ode_gradient(self, x, y):
        k = self.config.params
        S = y[0, :, 0]
        I = y[0, :, 1]
        R = y[0, :, 2]
        S_t = torch.gradient(S, spacing=(self.config.t_torch,))[0]
        I_t = torch.gradient(I, spacing=(self.config.t_torch,))[0]
        R_t = torch.gradient(R, spacing=(self.config.t_torch,))[0]
        f_S = S_t - (- self.config.params.beta * S * I)
        f_I = I_t - (self.config.params.beta * S * I - self.config.params.gamma * I)
        f_R = R_t - (self.config.params.gamma * I)
        return torch.cat((f_S.reshape([-1, 1]), f_I.reshape([-1, 1]), f_R.reshape([-1, 1])), 1), torch.cat((S_t.reshape([-1, 1]), I_t.reshape([-1, 1]), R_t.reshape([-1, 1])), 1)

    def loss(self, y, iteration=-1):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n, dy = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

        boundary_iteration = int(0.3 * self.config.args.iteration)  # 1.0 if self.config.boundary and iteration > boundary_iteration else 0.0
        loss3 = (1.0 if self.config.boundary and iteration > boundary_iteration else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]),
                           y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]),
                           self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))
        loss4 = (1.0 if self.config.cyclic else 0) * sum(
            [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])

        stable_period = 0.9
        stable_iteration = int(0.3 * self.config.args.iteration)
        loss5 = (1.0 if self.config.stable and iteration > stable_iteration else 0) * self.criterion(
            torch.abs(0.06 - torch.abs(dy[int(stable_period * self.config.T_N):, :])),
            0.06 - torch.abs(dy[int(stable_period * self.config.T_N):, :]))
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

    def forward(self, x):
        x1_new = self.fc1(x)
        x2_new = self.fc2(x)
        x3_new = self.fc3(x)
        x = torch.cat((x1_new, x2_new, x3_new), -1)
        return x


if __name__ == "__main__":
    run(Config, FourierModel, PINNModel)

