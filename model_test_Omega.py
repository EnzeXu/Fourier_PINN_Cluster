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



class TrainArgs:
    iteration = 50000  # 20000 -> 50000
    epoch_step = 5000  # 1000
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

        self.model_name = "SIRAged_Test_Omega"
        self.args = TrainArgs


class FourierModel(nn.Module):
    def __init__(self, config):
        super(FourierModel, self).__init__()
        self.config = config
        self.time_string = get_now_string()
        self.alpha = nn.Parameter(torch.rand(1))
        self.fc = nn.Linear(1, 1)
        self.criterion = torch.nn.MSELoss().to(self.config.device)  # "sum"

    def loss(self, y, iteration=-1):
        loss1 = self.criterion(y, torch.tensor([3.1415926]).to(self.config.device))

        loss = loss1
        loss_list = [loss1]
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
        lr_record = []

        for epoch in range(1, self.config.args.iteration + 1):
            optimizer.zero_grad()

            y = self.alpha
            loss, loss_list = self.loss(y, epoch)

            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)  # retain_graph=True
            optimizer.step()
            scheduler.step()

            now_time = time.time()
            time_record.append(now_time - start_time_0)
            lr_record.append(optimizer.param_groups[0]["lr"])

            if epoch % self.config.args.epoch_step == 0 or epoch == self.config.args.iteration:
                loss_print_part = " ".join(
                    ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
                myprint(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.12f} {3} NMSE-Loss: {4:.12f} Lr:{5:.12f} Time:{6:.6f}s ({7:.2f}min in total, {8:.2f}min remains)".format(
                        epoch, self.config.args.iteration, loss.item(), loss_print_part, loss.item(),
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                        (now_time - start_time_0) / 60.0 / epoch * (self.config.args.iteration - epoch)), self.config.args.log_path)
                start_time = now_time

                if epoch % self.config.args.test_step == 0:
                    print("parameter alpha: {}".format(self.alpha.item()))

                    # myprint(str(train_info), self.config.args.log_path)
        plt.figure(figsize=(16, 9))
        plt.plot(lr_record)
        plt.savefig("test/scheduler.png", dpi=400)
        plt.close()


if __name__ == "__main__":
    run(Config, FourierModel, FourierModel)

