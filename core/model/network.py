#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author: SnowPhoenix


import random
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from ..config import config
from ..utils import alloc_logger
import matplotlib.pyplot as plt
import sys
import os
import shutil

import torch
import torch.nn as nn

class BpNNRegressor(nn.Module):
    """
    神经网络回归器
    """

    def __init__(self, layer_sizes: tuple or list, hidden_activation=None):
        super(BpNNRegressor, self).__init__()
        self.logger = alloc_logger("BpNNRegressor.log", BpNNRegressor)

        if hidden_activation is None:
            hidden_activation = nn.Sigmoid

        layer_list = []  # 网络列表
        for i in range(len(layer_sizes) - 2):
            layer_list.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layer_list.append(hidden_activation())
        layer_list.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.layers = nn.Sequential(*layer_list)

        self.logger = alloc_logger("NNRegressor.log", BpNNRegressor)
        self.logger.log_message("CREATING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.logger.log_message("layers:\n", self)
        # self.logger.log_message("------------------------------------------------")

    def forward(self, X):
        return self.layers(X)
    



if __name__ == "__main__":
    network = BpNNRegressor([1, 20, 20, 1], hidden_activation=nn.ReLU)
    print("cuda_isavailable", torch.cuda.is_available())
    signature = "__main__"
    network.logger.log_message("------------- START SIMPLE TEST -------------", head=signature)

    range_num = 1
    iter_num = 1000
    error_rate_sum = 0
    network.logger.log_message("TOTAL RANGE:\t", range_num, head=signature)
    for i in range(range_num):
        network.logger.log_message("RANGE[", i + 1, "]:\t", head=signature)
        train_sample_size = 400
        network.logger.log_message("train-sample size:\t", train_sample_size, head=signature)
       
       
        train_samples = torch.unsqueeze(torch.linspace(-1, 1, train_sample_size), dim=1)
        train_labels = train_samples.pow(3) + 0.1 * torch.randn(train_samples.size())
        

        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        loss_func = nn.MSELoss()

        plt.ion()
        plt.show()

        for t in range(1000):
            x = train_samples
            y = train_labels

            prediction = network(x)
            loss = loss_func(prediction, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 5 == 0:
                plt.cla()
                plt.scatter(x.data.numpy(), y.data.numpy())
                plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
                plt.text(0.5, 0, 'Loss = %.4f' % loss.data,
                        fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.05)

            if loss < 1e-2:
                break
        
        plt.ioff()
        plt.show()


    network.logger.log_message("------------- FINISH SIMPLE TEST -------------", head=signature)


