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
        self.boundary_list = np.asarray([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]])

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
        return torch.cat((f_S.reshape([-1, 1]), f_I.reshape([-1, 1]), f_R.reshape([-1, 1])), 1)


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

