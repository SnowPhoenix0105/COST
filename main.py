import torch
import logging
from core.config import config

if __name__ == '__main__':
    print("shit")
    print(torch.cuda.is_available())
    print
    print("DATA\t", config.PATH.DATA)
    print("test.csv\t", config.PATH.DATA_TEST_CSV)
    print("test.csv\t", config.PATH.DATA_TEST_CSV)