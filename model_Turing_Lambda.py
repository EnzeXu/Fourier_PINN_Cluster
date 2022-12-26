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

from utils import *


class Parameters:
    N = 40
    M = 40
    d1 = 1
    d2 = 40
    c1 = 0.1  # 0.1
    c2 = 0.9  # 0.9
    c_1 = 1
    c3 = 1
    l = 0.6
    w = 0.6


class TrainArgs:
    iteration = 10#10000
    epoch_step = 1#500  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.001
    ignore_save_flag = True
    main_path = "."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config:
    def __init__(self):
        self.model_name = "Turing_Fourier_Lambda"
        self.curve_names = ["U", "V"]
        self.params = Parameters
        self.args = TrainArgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0
        self.layer = -1

        self.T_before = 30
        self.noise_rate = 0.05
        self.T = 2
        self.T_unit = 2e-3
        self.T_N_before = int(self.T_before / self.T_unit)
        self.T_N = int(self.T / self.T_unit)

        self.prob_dim = 2
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.y0_before = torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) + 2.0
        self.t_before = np.asarray([i * self.T_unit for i in range(self.T_N_before)])
        self.t = np.asarray([i * self.T_unit for i in range(self.T_N)])
        self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
        x = torch.zeros([1, self.T_N, self.params.N, self.params.M, 1]).to(self.device)
        self.x = FNO3d.get_grid(x.shape, x.device)
        truth_path = "saves/turing_truth.npy"
        if os.path.exists(truth_path) and not self.args.ignore_save_flag:
            self.truth = torch.tensor(np.load(truth_path), dtype=torch.float32).to(self.device)
            self.y0 = self.truth[0]
            print("Truth exists. Loading...")
        else:
            truth_before = torchdiffeq.odeint(self.pend, self.y0_before.cpu(), torch.tensor(self.t_before),
                                              method='euler').to(self.device)
            noise = (torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) - 0.5) * self.noise_rate
            self.y0 = torch.abs(truth_before[-1] * (1.0 + noise) + 0.2)
            self.truth = torchdiffeq.odeint(self.pend, self.y0.cpu(), torch.tensor(self.t), method='euler').to(
                self.device)
            # np.save(truth_path, self.truth.cpu().detach().numpy())
        print("y0:")
        self.draw_turing(self.y0)
        print("Truth:")
        print("Truth U: max={0:.6f} min={1:.6f}".format(torch.max(self.truth[:, :, :, 0]).item(),
                                                        torch.min(self.truth[:, :, :, 0]).item()))
        print("Truth V: max={0:.6f} min={1:.6f}".format(torch.max(self.truth[:, :, :, 1]).item(),
                                                        torch.min(self.truth[:, :, :, 1]).item()))
        self.draw_turing(self.truth[-1])

        self.modes1 = 12  # 8
        self.modes2 = 12
        self.modes3 = 12
        self.width = 32  # 20

        self.activation = ""
        self.activation_id = -1
        self.strategy = -1

    def pend(self, t, y):
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], 2])
        reaction_part[:, :, 0] = self.params.c1 - self.params.c_1 * y[:, :, 0] + self.params.c3 * (y[:, :, 0] ** 2) * y[:, :, 1]
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


class SpectralConv3d(nn.Module):
    def __init__(self, config):
        super(SpectralConv3d, self).__init__()

        self.config = config
        self.in_channels = self.config.width
        self.out_channels = self.config.width
        self.modes1 = self.config.modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = self.config.modes2
        self.modes3 = self.config.modes3

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.config.modes1, self.config.modes2,
                                    self.config.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.config.modes1, :self.config.modes2, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.config.modes1, :self.config.modes2, :self.config.modes3], self.weights1)
        out_ft[:, :, -self.config.modes1:, :self.config.modes2, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.config.modes1:, :self.config.modes2, :self.config.modes3], self.weights2)
        out_ft[:, :, :self.config.modes1, -self.config.modes2:, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.config.modes1, -self.config.modes2:, :self.config.modes3], self.weights3)
        out_ft[:, :, -self.config.modes1:, -self.config.modes2:, :self.config.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.config.modes1:, -self.config.modes2:, :self.config.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, config):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.config = config
        self.modes1 = self.config.modes1
        self.modes2 = self.config.modes2
        self.modes3 = self.config.modes3
        self.width = self.config.width
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.config)
        self.conv1 = SpectralConv3d(self.config)
        self.conv2 = SpectralConv3d(self.config)
        self.conv3 = SpectralConv3d(self.config)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block3 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block4 = ActivationBlock(self.config).to(self.config.device)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = grid

        # x = torch.cat((x, grid), dim=-1)
        # print("cp1", x.shape)
        x = self.fc0(x)
        # print("cp2", x.shape)
        x = x.permute(0, 4, 1, 2, 3)
        # print("cp3", x.shape)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        # print("cp4", x1.shape)
        x2 = self.w0(x)
        # print("cp5", x2.shape)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block1(x)
        # print("cp6", x.shape)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block2(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block3(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.activate_block4(x)
        # print("cp7", x.shape)
        # x = x[..., :-self.padding]
        # print("cp8", x.shape)
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        # print("cp9", x.shape)
        x = self.fc1(x)
        # print("cp10", x.shape)
        x = F.gelu(x)
        # print("cp11", x.shape)
        x = self.fc2(x)
        # print("cp12", x.shape)
        return x

    @staticmethod
    def get_grid(shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


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
        self.activate_weights_raw = nn.Parameter(torch.rand(len(self.activate_list)).to(self.config.device),
                                                 requires_grad=True)
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
            self.balance_weights = nn.Parameter(torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]).to(self.config.device),
                                                requires_grad=True)
        # print("self.activate_weights device = {}".format(self.activate_weights.device))
        # self.original_activation = nn.GELU()

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


# def penalty_func(x):
#     return 1 * (- torch.tanh((x - 0.004) * 300) + 1)

class FourierModel(nn.Module):
    def __init__(self, config):
        super(FourierModel, self).__init__()
        self.time_string = get_now_string()
        self.config = config
        self.setup_seed(self.config.seed)

        self.f_model = FNO3d(config)

        # self.fc0 = nn.Linear(1, self.config.width)  # input channel is 2: (a(x), x)
        # # self.layers = Layers(config=self.config, n=self.config.layer).to(self.config.device)
        # self.conv1 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv2 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv3 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv4 = SpectralConv1d(self.config).to(self.config.device)
        # self.cnn1 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.cnn2 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.cnn3 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.cnn4 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block3 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block4 = ActivationBlock(self.config).to(self.config.device)

        # self.fc1 = nn.Linear(self.config.width, self.config.fc_map_dim)
        # self.fc2 = nn.Linear(self.config.fc_map_dim, self.config.prob_dim)

        self.criterion = torch.nn.MSELoss().to(self.config.device)  # "sum"

        self.y_tmp = None
        self.epoch_tmp = None
        self.loss_record_tmp = None
        self.real_loss_record_tmp = None
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
        myprint("early stop: {}".format("On" if self.config.args.early_stop else "Off"), self.config.args.log_path)
        self.truth_loss()

    def truth_loss(self):
        y_truth = self.config.truth.reshape(
            [1, self.config.T_N, self.config.params.N, self.config.params.M, self.config.prob_dim])
        # print("y_truth max:", torch.max(y_truth))
        # print("y_truth min:", torch.min(y_truth))
        tl, tl_list = self.loss(y_truth)
        loss_print_part = " ".join(
            ["Loss_{0:d}:{1:.8f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(tl_list)])
        print("Ground truth has loss: Loss:{0:.8f} {1}".format(tl.item(), loss_print_part))

    #  MSE-loss of predicted value against truth
    def real_loss(self, y):
        real_loss = self.criterion(y[0, :, :], torch.tensor(self.config.truth[:, :]).to(self.config.device))
        return real_loss

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

    # def forward(self, x):
    #     x = self.fc0(x)
    #     x = x.permute(0, 2, 1)

    #     # x = self.layers(x)
    #     x1 = self.conv1(x)
    #     x2 = self.cnn1(x)
    #     x = x1 + x2
    #     x = self.activate_block1(x)

    #     x1 = self.conv2(x)
    #     x2 = self.cnn2(x)
    #     x = x1 + x2
    #     x = self.activate_block2(x)

    #     x1 = self.conv3(x)
    #     x2 = self.cnn3(x)
    #     x = x1 + x2
    #     x = self.activate_block3(x)

    #     x1 = self.conv4(x)
    #     x2 = self.cnn4(x)
    #     x = x1 + x2
    #     x = self.activate_block4(x)

    #     x = x.permute(0, 2, 1)
    #     x = self.fc1(x)
    #     x = nn.functional.gelu(x)
    #     x = self.fc2(x)
    #     return x

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def ode_gradient(self, y):
        # y: 1 * T_N * N * M * 2
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

        return y_t - y_t_theory

    def loss(self, y):
        y0_pred = y[0, 0]
        y0_true = self.config.y0

        ode_y = self.ode_gradient(y)
        zeros_nD = torch.zeros([self.config.T_N, self.config.params.N, self.config.params.M, self.config.prob_dim]).to(
            self.config.device)

        loss1 = 1 * self.criterion(y0_pred, y0_true)
        loss2 = 1e-4 * self.criterion(ode_y, zeros_nD)

        loss3 = 1 * (self.criterion(torch.abs(y - 0.1), y - 0.1) + self.criterion(torch.abs(6.5 - y), 6.5 - y))
        # loss4 = self.criterion(1e-3 / (y[0, :, :] ** 2 + 1e-10), zeros_nD)
        # self.criterion(1e-3 / (ode_1 ** 2 + 1e-10), zeros_1D) + self.criterion(1e-3 / (ode_2 ** 2 + 1e-10), zeros_1D) + self.criterion(1e-3 / (ode_3 ** 2 + 1e-10), zeros_1D)
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
        real_loss_record = []
        time_record = []

        for epoch in range(1, self.config.args.iteration + 1):
            optimizer.zero_grad()

            y = self.f_model(self.config.x)
            loss, loss_list = self.loss(y)
            loss_record.append(loss.item())
            real_loss = self.real_loss(y)
            real_loss_record.append(real_loss.item())

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
                                                         (now_time - start_time_0) / 60.0 / epoch * (
                                                                     self.config.args.iteration - epoch)),
                    self.config.args.log_path)
                start_time = now_time

                if epoch % self.config.args.test_step == 0:
                    self.y_tmp = y
                    self.epoch_tmp = epoch
                    self.loss_record_tmp = loss_record
                    self.real_loss_record_tmp = real_loss_record
                    self.time_record_tmp = time_record
                    self.test_model()
                    # save_path_loss = "{}/{}_{}_loss.npy".format(self.train_save_path_folder, self.config.model_name, self.time_string)
                    # np.save(save_path_loss, np.asarray(loss_record))

                    myprint("saving training info ...", self.config.args.log_path)
                    train_info = {
                        "model_name": self.config.model_name,
                        "seed": self.config.seed,
                        "layer": self.config.layer,
                        "activation": self.config.activation,
                        "activation_id": self.config.activation_id,
                        "strategy": self.config.strategy,
                        "epoch": self.config.args.iteration,
                        "epoch_stop": self.epoch_tmp,
                        "loss_length": len(loss_record),
                        "loss": np.asarray(loss_record),
                        "real_loss": np.asarray(real_loss_record),
                        "time": np.asarray(time_record),
                        "y_predict": y[0, :, :].cpu().detach().numpy(),
                        "y_truth": np.asarray(self.config.truth.cpu().detach().numpy()),
                        "y_shape": self.config.truth.cpu().detach().numpy().shape,
                        # "config": self.config,
                        "time_string": self.time_string,
                        "initial_lr": self.config.args.initial_lr,
                        # "weights_raw": np.asarray([
                        #     self.activate_block1.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block2.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block3.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block4.activate_weights_raw.cpu().detach().numpy(),
                        # ]),
                        # "weights": np.asarray([
                        #     self.activate_block1.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block2.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block3.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block4.activate_weights.cpu().detach().numpy(),
                        # ]),
                        # "balance_weights": np.asarray([
                        #     self.activate_block1.balance_weights.cpu().detach().numpy(),
                        #     self.activate_block2.balance_weights.cpu().detach().numpy(),
                        #     self.activate_block3.balance_weights.cpu().detach().numpy(),
                        #     self.activate_block4.balance_weights.cpu().detach().numpy(),
                        # ]),
                        # "sin_weight": np.asarray([
                        #     self.activate_block1.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block2.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block3.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block4.activates[0].omega.cpu().detach().numpy(),
                        # ]),
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
        # y_draw = self.y_tmp[0].cpu().detach().numpy().swapaxes(0, 1)
        # x_draw = self.config.t
        # y_draw_truth = self.config.truth.swapaxes(0, 1)
        # save_path = "{}/{}_{}_epoch={}.png".format(self.figure_save_path_folder, self.config.model_name,
        #                                            self.time_string, self.epoch_tmp)
        # draw_two_dimension(
        #     y_lists=np.concatenate([y_draw, y_draw_truth], axis=0),
        #     x_list=x_draw,
        #     color_list=self.default_colors[: 2 * self.config.prob_dim],
        #     legend_list=self.config.curve_names + ["{}_true".format(item) for item in self.config.curve_names],
        #     line_style_list=["solid"] * self.config.prob_dim + ["dashed"] * self.config.prob_dim,
        #     fig_title="{}_{}_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp),
        #     fig_size=(8, 6),
        #     show_flag=True,
        #     save_flag=True,
        #     save_path=save_path,
        #     save_dpi=300,
        #     legend_loc="center right",
        # )
        # print("Figure is saved to {}".format(save_path))
        # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])
        u_draw_all = self.y_tmp[0, :, :, :, 0].reshape(self.config.T_N,
                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        u_draw_all_truth = self.config.truth[:, :, :, 0].reshape(self.config.T_N,
                                                                 self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        v_draw_all = self.y_tmp[0, :, :, :, 1].reshape(self.config.T_N,
                                                       self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        v_draw_all_truth = self.config.truth[:, :, :, 1].reshape(self.config.T_N,
                                                                 self.config.params.N * self.config.params.M).cpu().detach().numpy().swapaxes(
            0, 1)[[10 * i for i in range(10)]]
        x_draw = self.config.t
        draw_n = len(u_draw_all)
        save_path_2D = "{}/{}_{}_epoch={}_2D.png".format(self.figure_save_path_folder, self.config.model_name,
                                                         self.time_string, self.epoch_tmp)

        m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 6), tight_layout_flag=True, show_flag=False, save_flag=True,
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
        u = self.y_tmp[0, :, :, :, 0].cpu().detach().numpy()
        v = self.y_tmp[0, :, :, :, 1].cpu().detach().numpy()
        u_last = u[-1]
        v_last = v[-1]
        u_true = self.config.truth[:, :, :, 0].cpu().detach().numpy()
        v_true = self.config.truth[:, :, :, 1].cpu().detach().numpy()
        u_last_true = u_true[-1]
        v_last_true = v_true[-1]
        save_path_map_all = "{}/{}_{}_epoch={}_map_all.png".format(self.figure_save_path_folder, self.config.model_name,
                                                                   self.time_string, self.epoch_tmp)
        save_path_map_pred_only = "{}/{}_{}_epoch={}_map_pred_only.png".format(self.figure_save_path_folder,
                                                                               self.config.model_name, self.time_string,
                                                                               self.epoch_tmp)
        m = MultiSubplotDraw(row=2, col=2, fig_size=(16, 14), tight_layout_flag=True, save_flag=True,
                             save_path=save_path_map_all)
        m.add_subplot_turing(
            matrix=u_last,
            v_max=u_last_true.max(),
            v_min=u_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.add_subplot_turing(
            matrix=v_last,
            v_max=v_last_true.max(),
            v_min=v_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.add_subplot_turing(
            matrix=u_last_true,
            v_max=u_last_true.max(),
            v_min=u_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_U_true".format(self.config.model_name, self.time_string))
        m.add_subplot_turing(
            matrix=v_last_true,
            v_max=v_last_true.max(),
            v_min=v_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_V_true".format(self.config.model_name, self.time_string))
        m.draw()

        m = MultiSubplotDraw(row=1, col=2, fig_size=(16, 7), tight_layout_flag=True, show_flag=False, save_flag=True,
                             save_path=save_path_map_pred_only)
        m.add_subplot_turing(
            matrix=u_last,
            v_max=u_last_true.max(),
            v_min=u_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.add_subplot_turing(
            matrix=v_last,
            v_max=v_last_true.max(),
            v_min=v_last_true.min(),
            fig_title_size=10,
            number_label_size=10,
            fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp))
        m.draw()

        # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])

    def write_finish_log(self):
        loss_average_length = 100
        with open("saves/record.txt", "a") as f:
            f.write(
                "{0}\t{1}\tseed={2}\t{3:.2f}min\titer={4}\tll={5:.12f}\tlrl={6:.12f}\tactivation={7}\tactivation_id={8}\tstrategy={9}\n".format(
                    self.config.model_name,
                    self.time_string,
                    self.config.seed,
                    self.time_record_tmp[-1] / 60.0,
                    self.config.args.iteration,
                    sum(self.loss_record_tmp[-loss_average_length:]) / loss_average_length,
                    sum(self.real_loss_record_tmp[-loss_average_length:]) / loss_average_length,
                    self.config.activation,
                    self.config.activation_id,
                    self.config.strategy,
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


def run(args):
    config = Config()
    config.seed = args.seed
    config.layer = args.layer
    config.activation = args.activation
    config.activation_id = args.activation_id
    config.strategy = args.strategy
    config.args.main_path = args.main_path
    config.args.log_path = args.log_path
    model = FourierModel(config).to(config.device)
    model.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    parser.add_argument("--main_path", default="./", help="main_path")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # parser.add_argument("--init", type=str, default="none", help="init: none/xavier_uniform/xavier_normal/kaiming_uniform/kaiming_normal")
    parser.add_argument("--activation", default="plan3", type=str, help="activation plan")
    parser.add_argument("--activation_id", type=int, default=-1, help="activation plan id (only used when activation = 'plan2')")
    parser.add_argument("--strategy", type=int, default=0, help="0=ones 1=fixed 2=adaptive")
    parser.add_argument("--layer", type=int, default=4, help="number of layer")
    opt = parser.parse_args()
    opt.overall_start = get_now_string()

    myprint("log_path: {}".format(opt.log_path), opt.log_path)
    myprint("cuda is available: {}".format(torch.cuda.is_available()), opt.log_path)
    try:
        run(opt)
    except Exception as e:
        print("[Error]", e)
