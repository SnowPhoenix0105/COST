import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ..utils import alloc_logger
from .network import BpNNRegressor
from .model import Model
import os
from ..config import config


class Trainer:
    def __init__(self, model, optimizer=None, loss_func=None, graph_interval = 20, log_interval = 100):
        self.logger = alloc_logger("Trainer.log", Trainer)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(model.network.parameters(), lr=0.1)
        self.loss_func = loss_func if loss_func is not None else nn.MSELoss()
        self.model = model
        self.graph_interval = graph_interval
        self.log_interval = log_interval


    def train(self, samples, labels, target_loss, max_iter, graph_show=True):
        cuda_enable = torch.cuda.is_available()

        samples = self.model.in_scaler.fit_transform(samples)
        labels = self.model.out_scaler.fit_transform(labels)

        if cuda_enable:
            samples = torch.from_numpy(samples).float().cuda()
            labels = torch.from_numpy(labels).float().cuda()

        if graph_show:
            plt.ion()
            plt.show()

        loss_list = []
        time_list = []

        network = self.model.network
        if cuda_enable:
            network = network.cuda()

        loss_sum = 0
        for t in range(max_iter):
            # shuffle
            idx = torch.randperm(samples.shape[0])
            if cuda_enable:
                idx = idx.cuda()
            x = samples[idx]
            y = labels[idx]

            prediction = network(x)
            loss = self.loss_func(prediction, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss)
            if graph_show and t % self.graph_interval == self.graph_interval - 1:
                time_list.append(t)
                loss_list.append(loss_sum / 5)
                loss_sum = 0
                plt.cla()
                plt.plot(time_list, loss_list, 'r-', lw=5)
                plt.text(max(time_list) / 2, max(loss_list) / 2, 'Loss = %.4f' % loss.data,
                        fontdict={'size': 10, 'color': 'red'})
                plt.pause(0.05)
            
            if t % self.log_interval == self.log_interval - 1:
                self.logger.log_message("[{:d}]loss=".format(t), float(loss))

            if loss < 5e-3:
                break

        img_file_name = os.path.join(config.PATH.IMAGE, self.logger.get_fs_legal_time_stampe() + ".png")
        try:
            dir_name = os.path.dirname(img_file_name)
            os.makedirs(dir_name)
            self.logger.log_message("save_graph_to_file():\tmakedirs: [", dir_name, ']')
        except FileExistsError:
            pass
        plt.cla()
        plt.plot(time_list, loss_list, 'r-', lw=5)
        plt.text(max(time_list) / 2, max(loss_list) / 2, 'last_loss = %.4f' % loss.data,
                fontdict={'size': 10, 'color': 'red'})
        plt.savefig(img_file_name)
        self.logger.log_message("save loss-down graph in [", img_file_name, ']')
        
        if graph_show:
            plt.ioff()
            plt.close()

        self.model.network = network.cpu()
        self.model.save()


class PTimesKFoldCrossValidation:
    def __init__(self, p, k, optimizer, loss_func):
        self.p = p
        self.k = k
        self.loss_func = loss_func
        self.optimizer = optimizer


    def test(self, model, test_samples, test_labels):
        prediction = model(test_samples)
        loss = self.loss_func(prediction, test_labels)
        return loss

    def pk_test(self, model, samples, labels):
        pass

    


if __name__ == "__main__":
    network = BpNNRegressor([1, 20, 20, 1], hidden_activation=nn.ReLU)
    print("cuda_isavailable", torch.cuda.is_available())
    signature = "__main__"

    model = Model(network)

    print(model)

    trainer = Trainer(model)

    train_sample_size = 400
    
    train_samples = torch.unsqueeze(torch.linspace(-1, 1, train_sample_size), dim=1)
    train_labels = train_samples.pow(3) + 0.1 * torch.randn(train_samples.size())
    

    trainer.train(
        samples=train_samples.numpy(), 
        labels=train_labels.numpy(), 
        max_iter=1000,
        target_loss=5e-2)

    ckpoint_dir = os.path.join(config.PATH.CKPOINT, alloc_logger().get_fs_legal_time_stampe())
    model.save(ckpoint_dir)