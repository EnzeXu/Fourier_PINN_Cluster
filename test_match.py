import torch
import random
import time
import numpy as np
import matplotlib.pyplot as plt


class Config:
    iteration = 10000000
    epoch_step = 2000
    test_step = epoch_step * 10
    initial_lr = 0.01


class MatchModel(torch.nn.Module):
    def __init__(self, config):
        super(MatchModel, self).__init__()
        self.setup_seed(0)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = torch.nn.Parameter(torch.tensor([0.0014706622, 1.5877504, 1.0, 0.0, 0.0, -0.0005020078, -0.022107242])) # [0.00023, 1.85, 0.0, -0.001, 1.28]
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.x_truth_numpy = np.asarray([3.0, 6.0, 9.0, 11.0, 12.0]) - 3.0
        self.x_truth = torch.tensor(self.x_truth_numpy, dtype=torch.float32).to(self.device)
        self.y_truth = torch.tensor(
            [1.280493662994179, 1.2930079443619786, 1.328899286950989, 1.387416558503003, 1.6444718895619048]).to(
            self.device) - 1.28
        self.x_plot = np.linspace(np.min(self.x_truth_numpy), np.max(self.x_truth_numpy), 1000)
        self.x_plot_torch = torch.tensor(self.x_plot).to(self.device)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def f(self, x, params):
        return params[0] * (params[1] ** (params[2] * x + params[3])) + params[5] * x + params[6]

    def loss(self):
        y_predict = self.f(self.x_truth, self.params)
        y_all = self.f(self.x_plot_torch, self.params)
        dy_all = torch.gradient(y_all, spacing=(self.x_plot_torch,))[0]
        loss1 = self.criterion(self.y_truth, y_predict)
        loss2 = 1e3 * self.criterion(torch.abs(y_all), y_all)
        loss3 = 1e3 * self.criterion(torch.abs(dy_all), dy_all)
        loss = loss1 + loss2 + loss3
        loss_list = [loss1, loss2, loss3]
        return loss, loss_list


    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.initial_lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 100000 + 1))
        self.train()

        start_time = time.time()
        start_time_0 = start_time

        for epoch in range(1, self.config.iteration + 1):
            optimizer.zero_grad()
            loss, loss_list = self.loss()
            loss.backward()
            optimizer.step()
            scheduler.step()

            now_time = time.time()

            if epoch % self.config.epoch_step == 0:
                loss_print_part = " ".join(
                    ["Loss_{0:d}:{1:.6f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(loss_list)])
                print(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.9f} {3} Lr:{4:.9f} Time:{5:.6f}s ({6:.2f}min in total, {7:.2f}min remains)".format(
                        epoch, self.config.iteration, loss.item(), loss_print_part,
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                                                         (now_time - start_time_0) / 60.0 / epoch * (
                                                                     self.config.iteration - epoch)))
                start_time = now_time

                if epoch % self.config.test_step == 0:
                    self.test_model()

    def test_model(self):
        # self.eval()
        params_numpy = self.params.detach().cpu().numpy()
        x_truth = self.x_truth.detach().cpu().numpy()
        y_truth = self.y_truth.detach().cpu().numpy()
        y_predict = self.f(self.x_truth, self.params).detach().cpu().numpy()
        x_plot = self.x_plot
        y_plot = self.f(x_plot, params_numpy)
        plt.figure(figsize=(8, 6))
        plt.plot(x_plot, y_plot, c="b")
        plt.scatter(x_truth, y_truth, c="r", label="truth")
        plt.scatter(x_truth, y_predict, c="b", label="predict")
        plt.legend()
        plt.show()
        # plt.close()
        print("params:", list(params_numpy))
        # self.train()


if __name__ == "__main__":
    config = Config
    model = MatchModel(config)
    model.train_model()