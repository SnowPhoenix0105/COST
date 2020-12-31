#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author: SnowPhoenix



from core.network.trainer import Trainer
import core.preprocessor.loader as loader
from core.utils import alloc_logger
from core.config import config
import os
import sys


logger = alloc_logger("Train.log", "Train")


def compare_certain_args(trainer, train_sample, train_label, *args):
    global logger

    # 对每组参数测试的次数
    loop_times = 8

    # 训练集占比
    train_rate = 0.7


    logger.log_message(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.log_message("args=", "\t".join(str(arg) for arg in args))
    trainer.reset(train_sample, train_label, *args)

    # 多次训练求平均
    error_rate_sum = 0
    time_cost_sum = 0
    for i in range(loop_times):
        # 训练
        _, error_rate, time_cost = trainer.train_and_test(train_rate)

        # 记录错误率
        error_rate_sum += error_rate
        time_cost_sum += time_cost
        logger.log_message("LOOP[{:d}]\t".format(i), "error_rate=", error_rate * 100, '%')
        
        # 绘制 loss 下降曲线
        trainer.network.draw_graph()
    
    # 计算平均性能    
    error_rate_mean = error_rate_sum / loop_times
    time_cost_mean = time_cost_sum / loop_times
    logger.log_message("TOTAL\terror_rate_mean=", error_rate_mean*100, "%\ttime_cost_mean=", time_cost_mean)
    
    # 保存 loss 下降曲线
    image_file_name = os.path.join(config.PATHS.IMAGE, trainer.network.logger.get_fs_legal_time_stampe() + ".png")
    trainer.network.save_graph_to_file(image_file_name)
    logger.log_message("save loss-down graph in file [", image_file_name, ']')


def compare_different_args():
    # 日志管理器
    global logger

    # 加载训练数据, 前者是训练输入, 后者是训练期望输出
    train_sample, train_label = loader.load_all_csv_as_ndarray(config.PATH.DATA_TRAIN_CSV)

    # 训练器
    trainer = Trainer(None, None, 0)

    # TODO 设置将要测试的参数集, 这些参数的笛卡尔积将作为神经网络的超参数传入.
    # 具体含义见 core/network/trainer.py 中 Trainer.reset() 方法的注释.
    # 可以通过 log/ 文件夹下的日志查看结果.
    hidden_layer_sizes_set = [(10,)] # 各个隐藏层大小
    eta_set = [1]               # 学习率
    eta_down_rate_set = [0.999]     # 学习率的下降率
    max_iter_set = [800]  # 最大训练迭代次数
    tol_set = [1e-2]      # 训练损失目标
    scaler_set = [None]         # 归一化器

    for size in hidden_layer_sizes_set:
        for eta in eta_set:
            for eta_down_rate in eta_down_rate_set:
                for max_iter in max_iter_set:
                    for tol in tol_set:
                        for scaler in scaler_set:
                            compare_certain_args(trainer, train_sample, train_label, size, eta, eta_down_rate, max_iter, tol, scaler)
        


def train():
    # 日志管理器
    global logger
    signature = "train():\t"

    # 加载训练数据, 前者是训练输入, 后者是训练期望输出
    train_sample, train_label = loader.load_all_csv_as_ndarray(config.PATH.DATA_TRAIN_CSV)

    # 训练器
    trainer = Trainer(None, None, 0)

    trainer.reset(
        sample=train_sample,
        label=train_label,
        # TODO 以下参数填写为 compare_different_args() 测试获得的最佳的参数.
        hidden_layer_sizes=(30,), 
        eta=1, 
        eta_down_rate=0.999,
        max_iter=800, 
        tol=0.01,
        scaler= None)

    network = trainer.train()

    logger.log_message(signature, "lase error=", network.last_error)
    ckpoint_dir = os.path.join(config.PATHS.CKPT, "final")
    logger.log_message(signature, "save ckpoint in dir [", ckpoint_dir, ']')
    network.save_to_file(ckpoint_dir)
    network.draw_graph()
    network.save_graph_to_file()

if __name__ == "__main__":
    # TODO 先通过调用 compare_different_args() 寻找最佳参数.
    # 然后注释掉对其的调用, 将 train() 调用的注释取消, 并将寻找到的最佳参数填写到 train() 中.
    compare_different_args()
    # train()