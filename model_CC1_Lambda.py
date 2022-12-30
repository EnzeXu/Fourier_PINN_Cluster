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
    MPF = 0.298
    Kinp = 0.558
    APCp = 0.591
    ksycycb = 0.04
    kde1cycb = 0.02
    kde2cycb = 0.4
    kphgwl = 0.2
    kdp1gwl = 0.08
    jphgwl = 0.01
    jdpgwl = 0.01
    kphapc = 0.2
    kdpapc = 0.08
    jphapc = 0.1
    jdpapc = 0.1


class TrainArgs:
    iteration = 10000
    epoch_step = 500  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.001
    main_path = "."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config:
    def __init__(self):
        self.model_name = "CC1_Fourier_Lambda"
        self.curve_names = ["MPF", "Kinp", "APCp"]
        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0
        self.layer = -1
        self.init = "none"
        self.pinn = 0

        self.T = 100
        self.T_unit = 2e-1
        self.T_N = int(self.T / self.T_unit)

        self.prob_dim = 3
        self.y0 = np.asarray([0.18140113, 0.27904593, 0.54060194])
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        self.x = torch.tensor(np.asarray([[[i * self.T_unit] * 1 for i in range(self.T_N)]]),
                              dtype=torch.float32).to(self.device)
        self.truth = odeint(self.pend, self.y0, self.t)

        self.modes = 64  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.width = 16
        self.fc_map_dim = 128

        self.activation = ""
        self.penalty = -1
        self.strategy = -1

    def pend(self, y, t):
        k = self.params
        dydt = np.asarray([
            k.ksycycb -  (k.kde1cycb + k.kde2cycb * y[2]) * y[0],
            k.kphgwl * y[0] * (1-y[1])/ (k.jphgwl + (1-y[1])) - k.kdp1gwl * y[1] / (k.jdpgwl + y[1]),
            k.kphapc * y[1] * (1- y[2]) / (k.jphapc + (1- y[2])) - k.kdpapc * y[2] / (k.jdpapc + y[2]),
        ])
        return dydt


class SpectralConv1d(nn.Module):
    def __init__(self, config):
        super(SpectralConv1d, self).__init__()
        self.config = config
        self.in_channels = self.config.width
        self.out_channels = self.config.width
        self.scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes, dtype=torch.cfloat))
        assert self.config.init in ["none", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]
        if self.config.init == "xavier_uniform":
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("relu"))
        elif self.config.init == "xavier_normal":
            nn.init.xavier_normal_(self.weights, gain=nn.init.calculate_gain("relu"))
        elif self.config.init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weights, nonlinearity="relu")
        elif self.config.init == "kaiming_normal":
            nn.init.kaiming_normal_(self.weights, nonlinearity="relu")
        else:
            pass

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


def get_now_string():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


class MySin(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor([1.4938150574984748]))

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
        self.activates = nn.ModuleList([activation_func(item).to(config.device) for item in self.activate_list])
        self.activate_weights_raw = nn.Parameter(torch.rand(len(self.activate_list)).to(self.config.device), requires_grad=True)
        # self.softmax = nn.Softmax(dim=0)

        # print("initial weights:", self.softmax(self.activate_weights_raw.clone()).detach().cpu().numpy())
        # self.softmax = nn.Softmax(dim=0).to(config.device)

        # self.activate_weights = self.softmax(self.activate_weights_raw)
        self.activate_weights = my_softmax(self.activate_weights_raw)
        assert self.config.strategy in [0, 1, 2]
        if self.config.strategy == 0:
            self.balance_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device)
        elif self.config.strategy == 1:
            self.balance_weights = torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device)
        else:
            self.balance_weights = nn.Parameter(torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device), requires_grad=True)
        # print("self.activate_weights device = {}".format(self.activate_weights.device))

    def forward(self, x):
        if self.config.activation == "original":
            return nn.functional.gelu(x)
        # print("now weights:", self.softmax(self.activate_weights_raw.clone()).detach().cpu().numpy())
        activation_res = 0.0
        for i in range(len(self.activate_list)):
            tmp_sum = self.activate_weights[i] * self.activates[i](x) * self.balance_weights[i]
            # print("{}: {} / ".format(self.activate_list[i], torch.mean(tmp_sum).item()), end="")
            activation_res += tmp_sum
        # print()
        # activation_res = sum([self.activate_weights[i] * self.activates[i](x) for i in range(len(self.activate_list))])
        return activation_res

#
# class BasicBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.conv = SpectralConv1d(self.config).to(self.config.device)
#         self.cnn = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
#         self.activate_block = ActivationBlock(self.config).to(self.config.device)
#
#     def forward(self, x):
#         x1 = self.conv(x)
#         x2 = self.cnn(x)
#         x = x1 + x2
#         x = self.activate_block(x)
#         return x
#
#
# class Layers(nn.Module):
#     def __init__(self, config, n=1):
#         super().__init__()
#         self.config = config
#         self.activate = ActivationBlock(self.config).to(self.config.device)
#         self.blocks = nn.Sequential(
#             *[BasicBlock(self.config).to(self.config.device) for _ in range(n)]
#         )
#
#     def forward(self, x):
#         x = self.blocks(x)
#         return x

def my_softmax(x):
    exponent_vector = torch.exp(x)
    sum_of_exponents = torch.sum(exponent_vector)
    softmax_vector = exponent_vector / sum_of_exponents
    return softmax_vector

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

        assert self.config.init in ["none", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]
        if self.config.init == "xavier_uniform":
            nn.init.xavier_uniform_(self.cnn1.weight, gain=1.0)
            nn.init.xavier_uniform_(self.cnn2.weight, gain=1.0)
            nn.init.xavier_uniform_(self.cnn3.weight, gain=1.0)
            nn.init.xavier_uniform_(self.cnn4.weight, gain=1.0)
            nn.init.xavier_uniform_(self.fc0.weight, gain=1.0)
            nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
            nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        elif self.config.init == "xavier_normal":
            nn.init.xavier_normal_(self.cnn1.weight, gain=1.0)
            nn.init.xavier_normal_(self.cnn2.weight, gain=1.0)
            nn.init.xavier_normal_(self.cnn3.weight, gain=1.0)
            nn.init.xavier_normal_(self.cnn4.weight, gain=1.0)
            nn.init.xavier_normal_(self.fc0.weight, gain=1.0)
            nn.init.xavier_normal_(self.fc1.weight, gain=1.0)
            nn.init.xavier_normal_(self.fc2.weight, gain=1.0)
        elif self.config.init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.cnn1.weight, nonlinearity="conv1d")
            nn.init.kaiming_uniform_(self.cnn2.weight, nonlinearity="conv1d")
            nn.init.kaiming_uniform_(self.cnn3.weight, nonlinearity="conv1d")
            nn.init.kaiming_uniform_(self.cnn4.weight, nonlinearity="conv1d")
            nn.init.kaiming_uniform_(self.fc0.weight, nonlinearity="linear")
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="linear")
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="linear")
        elif self.config.init == "kaiming_normal":
            nn.init.kaiming_normal_(self.cnn1.weight, nonlinearity="conv1d")
            nn.init.kaiming_normal_(self.cnn2.weight, nonlinearity="conv1d")
            nn.init.kaiming_normal_(self.cnn3.weight, nonlinearity="conv1d")
            nn.init.kaiming_normal_(self.cnn4.weight, nonlinearity="conv1d")
            nn.init.kaiming_normal_(self.fc0.weight, nonlinearity="linear")
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="linear")
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        else:
            pass

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
        self.default_colors = ["red", "blue", "green", "orange", "cyan", "purple", "pink", "indigo", "brown", "grey",
                               "indigo", "olive"]

        myprint("using {}".format(str(self.config.device)), self.config.args.log_path)
        myprint("iteration = {}".format(self.config.args.iteration), self.config.args.log_path)
        myprint("epoch_step = {}".format(self.config.args.epoch_step), self.config.args.log_path)
        myprint("test_step = {}".format(self.config.args.test_step), self.config.args.log_path)
        myprint("model_name = {}".format(self.config.model_name), self.config.args.log_path)
        myprint("time_string = {}".format(self.time_string), self.config.args.log_path)
        myprint("seed = {}".format(self.config.seed), self.config.args.log_path)
        myprint("num_layer = {}".format(self.config.layer), self.config.args.log_path)
        myprint("init = {}".format(self.config.init), self.config.args.log_path)
        myprint("early stop: {}".format("On" if self.config.args.early_stop else "Off"), self.config.args.log_path)
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
        MPF = y[0, :, 0]
        Kinp = y[0, :, 1]
        APCp = y[0, :, 2]
        MPF_t = torch.gradient(MPF, spacing=(self.config.t_torch,))[0].reshape([self.config.T_N])
        Kinp_t = torch.gradient(Kinp, spacing=(self.config.t_torch,))[0].reshape([self.config.T_N])
        APCp_t = torch.gradient(APCp, spacing=(self.config.t_torch,))[0].reshape([self.config.T_N])
        f_MPF = k.ksycycb -  (k.kde1cycb + k.kde2cycb * APCp) * MPF - MPF_t
        f_Kinp = k.kphgwl * MPF * (1 - Kinp)/ (k.jphgwl + (1 - Kinp))- k.kdp1gwl * Kinp / (k.jdpgwl + Kinp) - Kinp_t
        f_APCp = k.kphapc * Kinp * (1- APCp) / (k.jphapc + (1-APCp)) - k.kdpapc * APCp / (k.jdpapc + APCp) - APCp_t
        return torch.cat((f_MPF, f_Kinp, f_APCp))

    def loss(self, y):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N * self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 10 * (self.criterion(ode_n, zeros_nD))
        loss3 = self.criterion(torch.abs(y - 0), y - 0) + self.criterion(torch.abs(10 - y), 10 - y)
        y_norm = (y[0] - torch.min(y[0])) / (torch.max(y[0]) - torch.min(y[0]))
        loss4 = (1e-2 if self.config.penalty else 0) * torch.mean(penalty_func(torch.var(y_norm, dim=0)))
        # loss4 = self.criterion(1 / u_0, pt_all_zeros_3)
        # loss5 = self.criterion(torch.abs(u_0 - v_0), u_0 - v_0)

        loss = loss1 + loss2 + loss3 + loss4
        loss_list = [loss1, loss2, loss3, loss4]
        return loss, loss_list

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 200 + 1))
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
                        "layer": self.config.layer,
                        "init": self.config.init,
                        "activation": self.config.activation,
                        "penalty": self.config.penalty,
                        "strategy": self.config.strategy,
                        "epoch": self.config.args.iteration,
                        "epoch_stop": self.epoch_tmp,
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
                        "initial_lr": self.config.args.initial_lr,
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
                        "balance_weights": np.asarray([
                            self.activate_block1.balance_weights.cpu().detach().numpy(),
                            self.activate_block2.balance_weights.cpu().detach().numpy(),
                            self.activate_block3.balance_weights.cpu().detach().numpy(),
                            self.activate_block4.balance_weights.cpu().detach().numpy(),
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
        print("Figure is saved to {}".format(save_path))
        # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])

    def write_finish_log(self):
        loss_average_length = 1000
        with open("saves/record.txt", "a") as f:
            f.write(
                "{0}\t{1}\tseed={2}\t{3:.2f}min\titer={4}\tloss={5:.12f}\treal_loss_mse={6:.12f}\treal_loss_nmse={7:.12f}\tactivation={8}\tpenalty={9}\tstrategy={10}\tpinn={11}\tinit={12}\n".format(
                    self.config.model_name,
                    self.time_string,
                    self.config.seed,
                    self.time_record_tmp[-1] / 60.0,
                    self.config.args.iteration,
                    sum(self.loss_record_tmp[-loss_average_length:]) / loss_average_length,
                    sum(self.real_loss_mse_record_tmp[-loss_average_length:]) / loss_average_length,
                    sum(self.real_loss_nmse_record_tmp[-loss_average_length:]) / loss_average_length,
                    self.config.activation,
                    self.config.penalty,
                    self.config.strategy,
                    self.config.pinn,
                    self.config.init,
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


class PINNModel(nn.Module):
    def __init__(self, config):
        super(PINNModel, self).__init__()
        self.time_string = get_now_string()
        self.config = config
        self.config.model_name = self.config.model_name.replace("Fourier", "PINN")
        self.setup_seed(self.config.seed)

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
        self.default_colors = ["red", "blue", "green", "orange", "cyan", "purple", "pink", "indigo", "brown", "grey",
                               "indigo", "olive"]

        myprint("using {}".format(str(self.config.device)), self.config.args.log_path)
        myprint("iteration = {}".format(self.config.args.iteration), self.config.args.log_path)
        myprint("epoch_step = {}".format(self.config.args.epoch_step), self.config.args.log_path)
        myprint("test_step = {}".format(self.config.args.test_step), self.config.args.log_path)
        myprint("model_name = {}".format(self.config.model_name), self.config.args.log_path)
        myprint("time_string = {}".format(self.time_string), self.config.args.log_path)
        myprint("seed = {}".format(self.config.seed), self.config.args.log_path)
        myprint("num_layer = {}".format(self.config.layer), self.config.args.log_path)
        myprint("init = {}".format(self.config.init), self.config.args.log_path)
        myprint("early stop: {}".format("On" if self.config.args.early_stop else "Off"), self.config.args.log_path)
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
        x1_new = self.fc1(x)
        x2_new = self.fc2(x)
        x3_new = self.fc3(x)
        x = torch.cat((x1_new, x2_new, x3_new), -1)
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
        MPF = y[0, :, 0]
        Kinp = y[0, :, 1]
        APCp = y[0, :, 2]
        MPF_t = torch.gradient(MPF, spacing=(self.config.t_torch,))[0].reshape([self.config.T_N])
        Kinp_t = torch.gradient(Kinp, spacing=(self.config.t_torch,))[0].reshape([self.config.T_N])
        APCp_t = torch.gradient(APCp, spacing=(self.config.t_torch,))[0].reshape([self.config.T_N])
        f_MPF = k.ksycycb -  (k.kde1cycb + k.kde2cycb * APCp) * MPF - MPF_t
        f_Kinp = k.kphgwl * MPF * (1 - Kinp)/ (k.jphgwl + (1 - Kinp))- k.kdp1gwl * Kinp / (k.jdpgwl + Kinp) - Kinp_t
        f_APCp = k.kphapc * Kinp * (1- APCp) / (k.jphapc + (1-APCp)) - k.kdpapc * APCp / (k.jdpapc + APCp) - APCp_t
        return torch.cat((f_MPF, f_Kinp, f_APCp))

    def loss(self, y):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(self.config.x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N * self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 10 * (self.criterion(ode_n, zeros_nD))
        loss3 = self.criterion(torch.abs(y - 0), y - 0) + self.criterion(torch.abs(10 - y), 10 - y)
        # loss4 = self.criterion(1 / u_0, pt_all_zeros_3)
        # loss5 = self.criterion(torch.abs(u_0 - v_0), u_0 - v_0)

        loss = loss1 + loss2 + loss3
        loss_list = [loss1, loss2, loss3]
        return loss, loss_list

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 200 + 1))
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
                        "layer": self.config.layer,
                        "init": self.config.init,
                        "activation": self.config.activation,
                        "penalty": self.config.penalty,
                        "strategy": self.config.strategy,
                        "epoch": self.config.args.iteration,
                        "epoch_stop": self.epoch_tmp,
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
                        "initial_lr": self.config.args.initial_lr,
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
        print("Figure is saved to {}".format(save_path))
        # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])

    def write_finish_log(self):
        loss_average_length = 1000
        with open("saves/record.txt", "a") as f:
            f.write("{0}\t{1}\tseed={2}\t{3:.2f}min\titer={4}\tloss={5:.12f}\treal_loss_mse={6:.12f}\treal_loss_nmse={7:.12f}\tactivation={8}\tpenalty={9}\tstrategy={10}\tpinn={11}\tinit={12}\n".format(
                self.config.model_name,
                self.time_string,
                self.config.seed,
                self.time_record_tmp[-1] / 60.0,
                self.config.args.iteration,
                sum(self.loss_record_tmp[-loss_average_length:]) / loss_average_length,
                sum(self.real_loss_mse_record_tmp[-loss_average_length:]) / loss_average_length,
                sum(self.real_loss_nmse_record_tmp[-loss_average_length:]) / loss_average_length,
                self.config.activation,
                self.config.penalty,
                self.config.strategy,
                self.config.pinn,
                self.config.init,
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

def run(args, model_class):
    config = Config()
    config.seed = args.seed
    config.layer = args.layer
    config.init = args.init
    config.activation = args.activation
    config.penalty = args.penalty
    config.pinn = args.pinn
    config.strategy = args.strategy
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
    parser.add_argument("--init", type=str, default="none", help="init: none/xavier_uniform/xavier_normal/kaiming_uniform/kaiming_normal")
    parser.add_argument("--activation", default="plan3", type=str, help="activation plan")
    parser.add_argument("--penalty", type=int, default=1, help="0=off 1=on")
    parser.add_argument("--strategy", type=int, default=0, help="0=ones 1=fixed 2=adaptive")
    parser.add_argument("--layer", type=int, default=8, help="number of layer")
    opt = parser.parse_args()
    opt.overall_start = get_now_string()

    myprint("log_path: {}".format(opt.log_path), opt.log_path)
    myprint("cuda is available: {}".format(torch.cuda.is_available()), opt.log_path)
    try:
        if not opt.pinn:
            run(opt, FourierModel)
        else:
            run(opt, PINNModel)
    except Exception as e:
        print("[Error]", e)
