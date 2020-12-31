
import random
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from ..config import config
from ..utils import alloc_logger
from .network import BpNNRegressor
import matplotlib.pyplot as plt
import sys
import os
import shutil

import torch
import torch.nn as nn



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

    def train(self, X, out):
        #TODO
        pass