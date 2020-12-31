from .loader import origin_walk_csv
from ..config import config
from ..utils import alloc_logger

logger = alloc_logger("Detector.log")


def _detect_single_value_set(file_name):
    signature = "detect_value_set"
    set_age = set()
    set_sex = set()
    set_bmi = set()
    set_children = set()
    set_smoker = set()
    set_region = set()
    set_charges = set()
    count = 0
    for info in origin_walk_csv(file_name):
        set_age.add(info.age)
        set_sex.add(info.sex)
        set_bmi.add(info.bmi)
        set_children.add(info.children)
        set_smoker.add(info.smoker)
        set_region.add(info.region)
        set_charges.add(info.charges)
        count += 1
    global logger
    logger.log_message("total_num=", count, head=signature)
    logger.log_message("ages[{:d}]=".format(len(set_age)), set_age if len(
        set_age) < 10 else '', head=signature)
    logger.log_message("sexs[{:d}]=".format(len(set_sex)), set_sex if len(
        set_sex) < 10 else '', head=signature)
    logger.log_message("bmis[{:d}]=".format(len(set_bmi)), set_bmi if len(
        set_bmi) < 10 else '', head=signature)
    logger.log_message("childrens[{:d}]=".format(len(
        set_children)), set_children if len(set_children) < 10 else '', head=signature)
    logger.log_message("smokers[{:d}]=".format(len(set_smoker)), set_smoker if len(
        set_smoker) < 10 else '', head=signature)
    logger.log_message("regions[{:d}]=".format(len(set_region)), set_region if len(
        set_region) < 10 else '', head=signature)
    logger.log_message("chargess[{:d}]=".format(
        len(set_charges)), set_charges if len(set_charges) < 10 else '', head=signature)


def detect_value_set():
    signature = "detect_value_set"
    global logger
    logger.log_message("train-set:")
    _detect_single_value_set(config.PATH.DATA_TRAIN_CSV)
    logger.log_message("test-set:")
    _detect_single_value_set(config.PATH.DATA_TEST_CSV)

def detect_sample_num(csv_file_name)->int:
    return sum(1 for _ in origin_walk_csv(csv_file_name))

def detect_test_num()->int:
    return detect_sample_num(config.PATH.DATA_TEST_CSV)

def detect_train_num()->int:
    return detect_sample_num(config.PATH.DATA_TRAIN_CSV)
    

if __name__ == "__main__":
    detect_value_set()
    print(detect_test_num())
    print(detect_train_num())
