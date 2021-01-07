from core.preprocessor import loader
from core.config import config
import core.model as models
import os
import numpy as np
import torch
from train import train


# 标志是否需要重新训练，False表示从ckpoint加载
RETRAIN = True


if __name__ == "__main__":
    if RETRAIN:
        model = train([34], 10000)
    else:
        train_samples, train_labels = loader.load_all_csv_as_ndarray(config.PATH.DATA_TRAIN_CSV)
        network = models.BpNNRegressor([train_samples.shape[1], 20, 1], hidden_activation=torch.nn.ReLU)
        model = models.Model(network)
        model.load(os.path.join(config.PATH.CKPOINT, "pretrained"))

    samples, _ = loader.load_all_csv_as_ndarray(config.PATH.DATA_TEST_CSV)

    labels = model.predict(samples)

    with open(config.PATH.DATA_RESULT_CSV, 'w', encoding='utf8') as fout:
        with open(config.PATH.DATA_TEST_CSV, 'r', encoding='utf8') as fin:
            for row, line in enumerate(fin):
                if row != 0:
                    line = line.strip()[:-3] + str(labels[row - 1][0]) + '\n'
                fout.write(line)
            
    os.system("7z a -tzip " + config.PATH.DATA_RESULT_CSV.replace(".csv", '.zip') + " " + config.PATH.DATA_RESULT_CSV)


