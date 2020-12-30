#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author: SnowPhoenix



import numpy as np
import time
from ..utils import alloc_logger
from .bp_neural_network import BpNNClassifier
from ..config import config
import os


class DifferentSizeForDataAndLabel(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Trainer:
    def __init__(self,
            data: np.ndarray,
            teacher: np.ndarray,
            hidden_layer_sizes: tuple or list = (30, 30), 
            eta: float = 1, 
            eta_down_rate: float = 1,
            max_iter: float or int = 500, 
            tol: float = 1e-4,
            scaler: "one of sklearn.preprocessing.xxxScaler" = None):
        """
        训练器.

        @exception DifferentSizeForDataAndLabel: data 和 teacher 的第0维大小不一致.

        @param data: 输入数据;
        @param teacher: 期望输出;
        @param hidden_layer_sizes: 隐藏层的形状;
        @param eta: 学习率, 一般为1左右;
        @param eta_down_rate: 学习率的下降率, 每次训练迭代时 eta 变为 eta * eta_down_rate, 一般为0.99x;
        @param max_iter: 最大迭代次数, 当训练迭代次数到达此值后停止训练;
        @param tol: 目标精度, 当损失值低于此值并可以保持若干迭代时, 停止训练;
        @param scaler: 输入数据的归一化器, 默认采用MinMaxScaler;
        """
        super().__init__()
        self.data = None
        self.teacher = None
        self.hidden_layer_sizes = None
        self.eta = None
        self.eta_down_rate = None
        self.max_iter = None
        self.tol = None
        self.scaler = None
        self.network = None
        self.logger = alloc_logger("Trainer.log", Trainer)
        self.logger.log_message("CREATING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if data is not None:
            self.reset(
                data=data,
                teacher=teacher,
                hidden_layer_sizes=hidden_layer_sizes,
                eta=eta,
                eta_down_rate=eta_down_rate,
                max_iter=max_iter,
                tol=tol,
                scaler=scaler
            )

    def reset(self,
            data: np.ndarray,
            teacher: np.ndarray,
            hidden_layer_sizes: tuple or list = (30, 30), 
            eta: float = 1, 
            eta_down_rate: float = 1,
            max_iter: float or int = 500, 
            tol: float = 1e-4,
            scaler: "one of sklearn.preprocessing.xxxScaler" = None):
        """
        重新设置值.

        @exception DifferentSizeForDataAndLabel: data 和 teacher 的第0维大小不一致.

        @param data: 输入数据;
        @param teacher: 期望输出;
        @param hidden_layer_sizes: 神经网络隐藏层的形状;
        @param eta: 神经网络学习率, 一般为1左右;
        @param eta_down_rate: 神经网络学习率的下降率, 每次训练迭代时 eta 变为 eta * eta_down_rate, 一般为0.99x;
        @param max_iter: 神经网络最大迭代次数, 当训练迭代次数到达此值后停止训练;
        @param tol: 神经网络目标精度, 当损失值低于此值并可以保持若干迭代时, 停止训练;
        @param scaler: 神经网络输入数据的归一化器, 默认采用MinMaxScaler;
        """
        if data.shape[0] != teacher.shape[0]:
            raise DifferentSizeForDataAndLabel("data and teacher has different count")

        self.logger.log_message("set args:")
        self.logger.log_message("\tdata.shape=", data.shape)
        self.logger.log_message("\tteacher.shape=", teacher.shape)
        self.logger.log_message("\thidden_layer_sizes=", hidden_layer_sizes)
        self.logger.log_message("\teta=", eta)
        self.logger.log_message("\teta_down_rate=", eta_down_rate)
        self.logger.log_message("\tmax_iter=", max_iter)
        self.logger.log_message("\ttol=", tol)
        self.logger.log_message("\tscaler=", scaler)

        self.data = data
        self.teacher = teacher
        if (self.hidden_layer_sizes, self.eta, self.eta_down_rate, self.max_iter, self.tol, self.scaler) == (hidden_layer_sizes, eta, eta_down_rate, max_iter, tol, scaler):
            self.logger.log_message("network don't need to rebuild")
        else:
            self.logger.log_message("rebuilding network")
            self.hidden_layer_sizes = hidden_layer_sizes
            self.eta = eta
            self.eta_down_rate = eta_down_rate
            self.max_iter = max_iter
            self.tol = tol
            self.scaler = scaler
            self.network = BpNNClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                eta=eta,
                eta_down_rate=eta_down_rate,
                max_iter=max_iter,
                tol=tol,
                scaler=scaler
                )

    def train(self)->BpNNClassifier:
        signature = "train()\t"
        self.logger.log_message(signature, "start...")
        start_time = time.time()
        self.network.train(self.data, self.teacher)
        end_time = time.time()
        self.logger.log_message(signature, "finish!")
        self.logger.log_message("train()", "time_cost=" + str(end_time-start_time))
        return self.network
        
    def train_and_test(self, train_rate: float)->(int, float, float):
        """
        以 train_rate 作为输入数据和期望输出中, 用于训练的数据的比例;
        @param train_rate: 输入数据和期望输出中, 用于训练的数据的比例, 范围为(0, 1);
        @return: 错误数, 错误率, 总用时.
        """
        signature = "train_and_test()\t"
        self.logger.log_message(signature, "start...")
        self.logger.log_message(signature, "train_rate=", train_rate)
        total_count = self.data.shape[0]
        train_count = int(total_count * train_rate)
        test_count = total_count - train_count
        self.logger.log_message(signature, "total_count=", total_count)
        self.logger.log_message(signature, "train_count=", train_count)
        self.logger.log_message(signature, "test_count=", test_count)
        # 打乱
        idx = np.arange(total_count)
        np.random.shuffle(idx)
        self.data, self.teacher = self.data[idx], self.teacher[idx]
        # 开始
        start_time = time.time()
        self.network.train(self.data[:train_count], self.teacher[:train_count])
        predict_result = self.network.predict(self.data[train_count:])
        end_time = time.time()
        # 计算正确率
        bo = np.argwhere(predict_result != self.teacher[train_count:])
        count = len(bo) / 2
        error_rate = count / test_count
        self.logger.log_message(signature, "error_count=" + str(count))
        self.logger.log_message(signature, "error_rate=" + str(error_rate))
        self.logger.log_message(signature, "time_cost=" + str(end_time-start_time))
        self.logger.log_message(signature, "finish!")
        return count, error_rate, end_time - start_time


if __name__ == "__main__":
    trainer = Trainer(None, None, 0)
    signature = "__main__"

    total_size = 2000
    data = np.random.rand(total_size, 2) * 2 - 1
    teacher = np.zeros(shape=(total_size, 4))
    bo = data > 0
    teacher[:, 0] = (  bo[:, 0]  &   bo[:, 1]) * 1
    teacher[:, 1] = ((~bo[:, 0]) &   bo[:, 1]) * 1
    teacher[:, 2] = ((~bo[:, 0]) & (~bo[:, 1])) * 1
    teacher[:, 3] = (  bo[:, 0]  & (~bo[:, 1])) * 1

    trainer.reset(
        data=data, 
        teacher=teacher, 
        hidden_layer_sizes=(8, 4), 
        eta=1, 
        eta_down_rate=1, 
        max_iter=1000, 
        tol=1e-3, 
        scaler=None)

    total_range = 4
    trainer.logger.log_message("TOTAL_RANGE=", total_range, head=signature)

    total_count = 0
    total_error_rate = 0
    total_time_cost = 0
    for i in range(total_range):
        trainer.logger.log_message("RANGE [{:d}]".format(i), head=signature)

        count, error_rate, time_cost = trainer.train_and_test(0.2)

        file_dir = os.path.join(config.PATH.CKPOINT, "debug")
        file_dir = os.path.join(file_dir, trainer.network.logger.get_fs_legal_time_stampe())
        
        trainer.network.save_to_file(file_dir)
        image_file_name = os.path.join(config.PATH.IMAGE, "debug")
        image_file_name = os.path.join(image_file_name, trainer.network.logger.get_fs_legal_time_stampe() + ".png")
        trainer.network.save_graph_to_file(image_file_name)

        total_count += count
        total_error_rate += error_rate
        total_time_cost += time_cost
    
    trainer.logger.log_message("average error_count=", total_count / total_range, head=signature)
    trainer.logger.log_message("average error_rate=", total_error_rate / total_range * 100, '%', head=signature)
    trainer.logger.log_message("average time_cost=", total_time_cost / total_range, head=signature)
    