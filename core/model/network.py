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

    def save_to_file(self, file_dir: str = None):
        """
        @param file_dir: 将向该目录下写入保存文件, 会先清空该目录下所有文件和子目录
        """
        signature = "save_to_file():\t"
        if file_dir is None:
            file_dir = os.path.join(config.PATH.CKPT, self.logger.get_fs_legal_time_stampe())

        try:
            shutil.rmtree(file_dir)
            self.logger.log_message(signature, "clean up dir [", file_dir, ']')
        except FileNotFoundError:
            self.logger.log_message(signature, "makedirs: [", file_dir, ']')
        os.makedirs(file_dir)

        json_info = {}
        json_info["hidden_layer_sizes"] = self.hidden_layer_sizes
        json_info["eta"] = self.eta
        json_info["max_iter"] = self.max_iter
        json_info["tol"] = self.tol
        json_info["last_error"] = self.last_error
        json_info["eta_down_rate"] = self.eta_down_rate
        json_info["error_down_count"] = self.error_down_count
        json_info["error_down_data"] = self.error_down_data
        json_info["interval"] = self.interval
        json_info["W_list_count"] = len(self.W_list)
        with open(os.path.join(file_dir, "network.json"), 'w', encoding='utf-8') as f:
            json.dump(json_info, f)
        joblib.dump(self.in_scaler, os.path.join(file_dir, "in_scaler"))
        joblib.dump(self.out_scaler, os.path.join(file_dir, "out_scaler"))
        for i in range(len(self.W_list)):
            np.save(os.path.join(file_dir, str(i) + ".npy"), self.W_list[i])
        self.logger.log_message(signature, "save all status to dir [", file_dir, "]")

    def draw_graph(self):
        """
        绘图
        """
        X = np.array(self.error_down_count)
        Y = np.array(self.error_down_data)
        plt.plot(X, Y)
        self.logger.log_message("draw_graph():\tdraw loss-rate graph on canvas")

    def save_graph_to_file(self, file_name: str = None):
        """
        @param file_dir: 保存图片的文件名
        """
        if file_name is None:
            file_name = os.path.join(config.PATH.IMAGE, self.logger.get_fs_legal_time_stampe() + ".png")
        try:
            dir_name = os.path.dirname(file_name)
            os.makedirs(dir_name)
            self.logger.log_message("save_graph_to_file():\tmakedirs: [", dir_name, ']')
        except FileExistsError:
            pass
        plt.savefig(file_name)
        plt.cla()
        self.logger.log_message("save_graph_to_file():\tsave loss-rate graph in file [", file_name, ']')
        # plt.show()

    @staticmethod
    def load_from_file(file_dir: str):
        tmp_logger = alloc_logger()
        tmp_logger.log_message("load from dir [", file_dir, ']')
        with open(os.path.join(file_dir, "network.json"), 'r', encoding='utf-8') as f:
            json_info = json.load(f)
        ret = BpNNRegressor(
            hidden_layer_sizes=json_info["hidden_layer_sizes"],
            eta=json_info["eta"],
            eta_down_rate=json_info["eta_down_rate"],
            max_iter=json_info["max_iter"],
            tol=json_info["tol"]
        )
        ret.W_list = []
        ret.in_scaler = joblib.load(os.path.join(file_dir, "in_scaler"))
        ret.out_scaler = joblib.load(os.path.join(file_dir, "out_scaler"))
        for i in range(int(json_info["W_list_count"])):
            arr = np.load(os.path.join(file_dir, str(i) + ".npy"))
            ret.W_list.append(arr)
        return ret



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


