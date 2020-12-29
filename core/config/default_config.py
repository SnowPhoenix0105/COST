import os
import logging

class DefaultConfig:
    class PATH:
        TOP = os.getcwd()
        DATA = os.path.join(TOP, "data")
        DATA_PUBLIC = os.path.join(DATA, "public_dataset")
        DATA_TRAIN_CSV = os.path.join(DATA_PUBLIC, "train.csv")
        DATA_TEST_CSV = os.path.join(DATA_PUBLIC, "test_sample.csv")

    class TRAIN:
        BATCH_SIZE = 128

    class LOG:
        FILE_PATH = os.path.join(DefaultConfig.PATH.TOP, "log")

def auto_init():
    # TODO
    pass