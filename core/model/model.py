
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from ..config import config
from ..utils import alloc_logger
import os
import torch
import shutil
from .network import BpNNRegressor



class Model:
    def __init__(self, 
        network: BpNNRegressor,
        in_scaler: "one of sklearn.preprocessing.xxxScaler" = None,
        out_scaler : "one of sklearn.preprocessing.xxxScaler" = None):

        self.in_scaler = MinMaxScaler() if in_scaler is None else in_scaler      # 正则化器
        self.out_scaler = MinMaxScaler() if out_scaler is None else out_scaler      # 正则化器
        self.network = network                                                  # 神经网络
        
        self.logger = alloc_logger("NNRegressor.log", BpNNRegressor)
        self.logger.log_message("CREATING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.logger.log_message("\tin_scaler=", self.in_scaler)
        self.logger.log_message("\tout_scaler=", self.out_scaler)
        self.logger.log_message("\tnetwork=", self.network)
        # self.logger.log_message("------------------------------------------------")

    def __str__(self):
        return "Model[\nin_scaler={:},\nout_scaler={:},\nnetwork={}]".format(self.in_scaler, self.out_scaler, self.network)


    def __call__(self, X):
        return self.predict(X)

    def predict(self, X):
        X = self.in_scaler.transform(X)
        X = torch.from_numpy(X).float()

        Y = self.network(X)
        Y = Y.detach().numpy()
        return self.out_scaler.inverse_transform(Y)

    def save(self, file_dir=None):
        """
        @param file_dir: 将向该目录下写入保存文件, 会先清空该目录下所有文件和子目录
        """
        signature = "save():\t"
        if file_dir is None:
            file_dir = os.path.join(config.PATH.CKPOINT, self.logger.get_fs_legal_time_stampe())
        try:
            shutil.rmtree(file_dir)
            self.logger.log_message(signature, "clean up dir [", file_dir, ']')
        except FileNotFoundError:
            self.logger.log_message(signature, "makedirs: [", file_dir, ']')
        os.makedirs(file_dir)
        
        joblib.dump(self.in_scaler, os.path.join(file_dir, "in_scaler"))
        joblib.dump(self.out_scaler, os.path.join(file_dir, "out_scaler"))
        torch.save(self.network.state_dict(), os.path.join(file_dir, "network"))

        self.logger.log_message("save in [", file_dir, ']')

        return file_dir
    
    def load(self, file_dir):
        self.in_scaler = joblib.load(os.path.join(file_dir, "in_scaler"))
        self.out_scaler = joblib.load(os.path.join(file_dir, "out_scaler"))
        self.network.load_state_dict(torch.load(os.path.join(file_dir, "network")))
        self.logger.log_message("load from [", file_dir, ']')
        return self