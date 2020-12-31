import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ..utils import alloc_logger
from .network import BpNNRegressor
from .model import Model
import os
from ..config import config


class Trainer:
    def __init__(self, model, optimizer=None, loss_func=None, graph_interval = 10, log_interval = 50):
        self.logger = alloc_logger("Trainer.log", Trainer)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(network.parameters(), lr=0.1)
        self.loss_func = loss_func if loss_func is not None else nn.MSELoss()
        self.model = model
        self.graph_interval = graph_interval
        self.log_interval = log_interval


    def train(self, samples, labels, target_loss, max_iter):
        samples = self.model.in_scaler.fit_transform(samples)
        labels = self.model.out_scaler.fit_transform(labels)

        samples = torch.from_numpy(samples)
        labels = torch.from_numpy(labels)

        plt.ion()
        plt.show()

        loss_list = []
        time_list = []

        network = self.model.network

        loss_sum = 0
        for t in range(max_iter):
            # shuffle
            idx = torch.randperm(samples.shape[0])
            x = samples[idx]
            y = labels[idx]

            prediction = network(x)
            loss = self.loss_func(prediction, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss)
            if t % self.graph_interval == self.graph_interval - 1:
                time_list.append(t)
                loss_list.append(loss_sum / 5)
                loss_sum = 0
                plt.cla()
                plt.plot(time_list, loss_list, 'r-', lw=5)
                plt.text(max(time_list) / 2, max(loss_list) / 2, 'Loss = %.4f' % loss.data,
                        fontdict={'size': 10, 'color': 'red'})
                plt.pause(0.05)
            
            if t % self.log_interval == self.log_interval - 1:
                self.logger.log_message("loss=", float(loss))

            if loss < 5e-3:
                break

        img_file_name = os.path.join(config.PATH.IMAGE, self.logger.get_fs_legal_time_stampe() + ".png")
        try:
            dir_name = os.path.dirname(img_file_name)
            os.makedirs(dir_name)
            self.logger.log_message("save_graph_to_file():\tmakedirs: [", dir_name, ']')
        except FileExistsError:
            pass
        plt.savefig(img_file_name)
        self.logger.log_message("save loss-down graph in [", img_file_name, ']')
            
        plt.ioff()
        plt.close()

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
    model.load(ckpoint_dir)