#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author: SnowPhoenix


import random
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from ..config import config
from ..utils import alloc_logger
import matplotlib.pyplot as plt
import sys
import os
import shutil

class BpNNClassifier:
    """
    神经网络分类器
    """

    def __init__(self, 
            hidden_layer_sizes: tuple or list = (30, 30), 
            eta: float = 1, 
            eta_down_rate: float = 1,
            max_iter: float or int = 500, 
            tol: float = 1e-4,
            scaler: "one of sklearn.preprocessing.xxxScaler" = None
        ):

        self.hidden_layer_sizes = hidden_layer_sizes  # 各隐藏节点的个数
        self.eta = eta  # 随机梯度下降的学习率
        self.max_iter = max_iter  # 随机梯度下降最大迭代次数
        self.tol = tol  # 误差阈值
        self.last_error = 1
        self.eta_down_rate = eta_down_rate  # 每次迭代后eta的下降率
        self.W_list = []  # 矩阵列表
        self.scaler = MinMaxScaler() if scaler is None else scaler      # 正则化器
        self.interval = 50
        self.error_down_count = []
        self.error_down_data = []           # 错误率下降记录
        
        self.logger = alloc_logger("NNClassifier.log", BpNNClassifier)
        self.logger.log_message("CREATING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.logger.log_message("\thidden_layer_sizes=", self.hidden_layer_sizes)
        self.logger.log_message("\teta=", self.eta)
        self.logger.log_message("\teta_down_rate=", self.eta_down_rate)
        self.logger.log_message("\tmax_iter=", self.max_iter)
        self.logger.log_message("\ttol=", self.tol)
        self.logger.log_message("\tscaler=", self.scaler)
        # self.logger.log_message("------------------------------------------------")

    def _sigmoid(self, y: np.ndarray):
        """激活函数"""
        return 1. / (1. + np.exp(-y))

    def _y(self, x: np.ndarray, W: np.ndarray):
        """加权求和，计算节点净输入"""
        return np.matmul(x, W)

    def _error(self, out: np.ndarray, out_predict: np.ndarray):
        """计算误差"""
        return np.sum((out - out_predict) ** 2) / len(out)

    def _back_propagation(self, X: np.ndarray, out: np.ndarray):
        """
        反向传播（梯度下降）
        @param X: 输入
        @param out: 期望输出
        @return: 训练好的W_list
        """
        m, n = X.shape          # 训练样本数量，输入层维度
        _, n_z = out.shape      # _，输出层维度

        # 获得各层节点个数元组layer_sizes以及总层数layer_n
        layer_sizes = self.hidden_layer_sizes + (n_z,)
        layer_n = len(layer_sizes)

        # 对于每一层，将所有节点的权向量（以列向量形式）存为一个矩阵，保存至W_list
        W_list = []  # 矩阵列表
        dW_list = []
        li_size = n
        for lj_size in layer_sizes:     # li_size、lj_size为第i、j层节点数量
            W = np.random.randn(li_size + 1, lj_size) * 0.05
            # W_list[i]为第 i 层与第 j 层之间的权重矩阵，
            #   其中输入层为第0层，隐藏层为第1~layer_n-1层，输出层为第layer_n层
            W_list.append(W)            
            dW_list.append(np.zeros(shape=(li_size + 1, lj_size)))
            li_size = lj_size

        # 创建运行梯度下降时所使用的列表
        in_list = [None] * layer_n  # in_list[i]为第i层与第j层之间的输入
        y_list = [None] * layer_n   # y_list[i]为第i层与第j层之间的净输入
        z_list = [None] * layer_n   # y_list[i]为第i层与第j层的输出值，即作为下一层的输入
        delta_list = [None] * layer_n

        # 随机梯度下降
        count = 0
        end_count = 0
        self.error_down_data = []
        self.error_down_count = []
        error_sum = 0
        idx = np.arange(m)
        for _ in range(self.max_iter):
            # 随机打乱训练集
            np.random.shuffle(idx)
            X, out = X[idx], out[idx]

            for x, t in zip(X, out):
                # 单个样本作为输入，运行神经网络
                z = x
                for i in range(layer_n):
                    # 第i-1层输出添加z0=1， 作为第i层输入
                    in_ = np.ones(z.size + 1)
                    in_[1:] = z
                    # 计算第i层所有节点的净输入
                    y = self._y(in_, W_list[i])
                    # 计算第i层各节点输出值
                    z = self._sigmoid(y)
                    # 保存第i层各节点的输入，净输入，输出
                    in_list[i], y_list[i], z_list[i] = in_, y, z

                # 反向传播计算各层节点delta

                # 输出层
                delta_list[-1] = z * (1. - z) * (t - z)
                # 隐藏层
                for i in range(layer_n - 2, -1, -1):
                    z_i, W_j, delta_j = z_list[i], W_list[i + 1], delta_list[i + 1]
                    delta_list[i] = z_i * (1. - z_i) * np.matmul(W_j[1:], delta_j[:, None].T[0])

                # 更新所有节点的权
                for i in range(layer_n):
                    in_i, delta_i = in_list[i], delta_list[i]
                    dW_list[i] = in_i[:, None] * delta_i * self.eta
                    W_list[i] += dW_list[i]

            # 计算训练误差
            out_pred = self._predict(X, W_list)
            self.last_error = self._error(out, out_pred)

            # 记录错误率
            count += 1
            error_sum += self.last_error
            if count % self.interval == 0:
                self.error_down_count.append(count)
                self.error_down_data.append(error_sum / self.interval)
                self.logger.log_message("[{:d}]\t".format(count), "error=", error_sum / self.interval)
                error_sum = 0
                # print('.', end='')
                sys.stdout.flush()
                

            # 更新eta
            self.eta = self.eta * self.eta_down_rate

            # 判断收敛
            if self.last_error < self.tol:
                end_count += 1
                if end_count > 5:
                    break
            else:
                end_count = 0

        # print('')
        self.logger.log_message("loop=", count)
        self.logger.log_message("last_error=" + str(self.last_error))
        # 返回训练好的权矩阵列表
        return W_list

    def train(self, X: np.ndarray, out: np.ndarray):
        """
        训练
        @param X: 输入
        @param out: 期望输出
        @return: None
        """
        
        self.logger.log_message("start training...")
        
        # 归一化
        X = self.scaler.fit_transform(X)

        # 调用反向传播算法训练神经网络中所有节点的权
        self.W_list = self._back_propagation(X, out)

        self.logger.log_message("finish training!")

    def _predict(self, X: np.ndarray, W_list: list, return_bin: bool = False):
        """
        预测内部接口
        @param X: 输入
        @param W_list: 矩阵列表
        @param return_bin: 是否需要将输出转化为独热码形式
        @return: 输出
        """

        layer_n = len(W_list)

        z = X
        for i in range(layer_n):
            # 第i-1层输入添加x0=1，作为第i层输入
            m, n = z.shape
            in_ = np.ones((m, n + 1))
            in_[:, 1:] = z
            # 计算第i层所有节点的净输入
            y = self._y(in_, W_list[i])
            # 计算第i层所有节点的净输出值
            z = self._sigmoid(y)

        # 返回二进制编码的类标记
        if return_bin:
            # 输出最大节点输出编码为1， 其他节点输出编码为0
            idx = np.argmax(z, axis=1)
            z_bin = np.zeros_like(z)
            z_bin[range(len(idx)), idx] = 1
            return z_bin

        return z

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        @param X: 输入
        @return: 预测结果
        """
        X = self.scaler.transform(X)
        return self._predict(X, self.W_list, return_bin=True)

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
        joblib.dump(self.scaler, os.path.join(file_dir, "scaler"))
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
        ret = BpNNClassifier(
            hidden_layer_sizes=json_info["hidden_layer_sizes"],
            eta=json_info["eta"],
            eta_down_rate=json_info["eta_down_rate"],
            max_iter=json_info["max_iter"],
            tol=json_info["tol"]
        )
        ret.W_list = []
        ret.scaler = joblib.load(os.path.join(file_dir, "scaler"))
        for i in range(int(json_info["W_list_count"])):
            arr = np.load(os.path.join(file_dir, str(i) + ".npy"))
            ret.W_list.append(arr)
        return ret



if __name__ == "__main__":
    network = BpNNClassifier(
        hidden_layer_sizes=(4,), 
        eta=1, 
        eta_down_rate=1, 
        max_iter=1000, 
        tol=1e-3)
    signature = "__main__"
    network.logger.log_message("------------- START SIMPLE TEST -------------", head=signature)

    range_num = 4
    error_rate_sum = 0
    network.logger.log_message("TOTAL RANGE:\t", range_num, head=signature)
    for i in range(range_num):
        network.logger.log_message("RANGE[", i + 1, "]:\t", head=signature)
        train_sample_size = 400
        network.logger.log_message("train-sample size:\t", train_sample_size, head=signature)
        train_samples = np.random.rand(train_sample_size, 2) * 2 - 1
        train_teacher = np.zeros(shape=(train_sample_size, 4))
        bo = train_samples > 0
        train_teacher[:, 0] = (  bo[:, 0]  &   bo[:, 1]) * 1
        train_teacher[:, 1] = ((~bo[:, 0]) &   bo[:, 1]) * 1
        train_teacher[:, 2] = ((~bo[:, 0]) & (~bo[:, 1])) * 1
        train_teacher[:, 3] = (  bo[:, 0]  & (~bo[:, 1])) * 1

        network.train(train_samples, train_teacher)

        test_sample_size = 2000
        network.logger.log_message("test-sample size:\t", test_sample_size, head=signature)

        test_samples = np.random.rand(test_sample_size,2) * 2 - 1
        test_expect = np.zeros(shape=(test_sample_size, 4))
        bo = test_samples > 0
        test_expect[:, 0] = (  bo[:, 0]  &   bo[:, 1]) * 1
        test_expect[:, 1] = ((~bo[:, 0]) &   bo[:, 1]) * 1
        test_expect[:, 2] = ((~bo[:, 0]) & (~bo[:, 1])) * 1
        test_expect[:, 3] = (  bo[:, 0]  & (~bo[:, 1])) * 1
        test_answer = network.predict(test_samples)
        bo = test_answer != test_expect
        boo = bo[:, 0] | bo[:, 1] | bo[:, 2] | bo[:, 3]

        count = np.sum(boo)
        error_rate = count / test_sample_size
        error_rate_sum += error_rate
        network.logger.log_message("error count:\t", count, head=signature)
        network.logger.log_message("error rate:\t{:.2f}%".format(error_rate * 100), head=signature)

        # 保存ckpoint
        file_dir = os.path.join(config.PATH.CKPOINT, "debug")
        file_dir = os.path.join(file_dir, network.logger.get_fs_legal_time_stampe())
        network.save_to_file(file_dir)
        # 绘制 loss 下降曲线
        network.draw_graph()

        load_net_work = BpNNClassifier.load_from_file(file_dir)
        load_test_answer = load_net_work.predict(test_samples)
        is_same = (load_test_answer == test_answer).all()
        assert is_same
        network.logger.log_message("reload is same=", is_same, head=signature)

    # 保存 loss 下降曲线
    image_file_name = os.path.join(config.PATH.IMAGE, "debug")
    image_file_name = os.path.join(image_file_name, network.logger.get_fs_legal_time_stampe() + ".png")
    network.save_graph_to_file(image_file_name)


    network.logger.log_message("average error rate:\t{:.2f}%".format(error_rate_sum / range_num * 100), head=signature)

    network.logger.log_message("------------- FINISH SIMPLE TEST -------------", head=signature)


