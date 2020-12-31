from ..utils import alloc_logger
from ..config import config
from enum import IntEnum
import numpy as np

logger = alloc_logger("Loader.log")

class Regions(IntEnum):
    SouthEast = 0
    SouthWest = 1
    NorthEast = 2
    NorthWest = 3

Regions_to_str = {
    Regions.SouthEast: "southeast",
    Regions.SouthWest: "southwest",
    Regions.NorthEast: "northeast",
    Regions.NorthWest: "northwest"
}

str_to_Regions = {
    "southeast": Regions.SouthEast,
    "southwest": Regions.SouthWest,
    "northeast": Regions.NorthEast,
    "northwest": Regions.NorthWest
}

class Info:
    def __init__(self, age, sex, bmi, children, smoker, region, charges, is_raw: bool = False):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
        self.charges = charges
        self._is_raw = is_raw

    def is_raw(self):
        return self._is_raw

    def copy(self):
        return Info(self.age, self.sex, self.bmi, self.children, self.smoker, self.region, self.charges, self._is_raw)

    def to_format(self, copy: bool=False, check: bool = True):
        ret = self if not copy else self.copy()
        if not ret._is_raw:
            if check:
                raise Exception("call to_foramt() on formatted Info object")
            return ret
        ret.age = int(ret.age)
        ret.sex = True if ret.sex == "male" else False   # True 男, False 女
        ret.bmi = float(ret.bmi)
        ret.children = int(ret.children)
        ret.smoker = True if ret.smoker == 'yes' else False
        ret.region = str_to_Regions[ret.region]
        ret.charges = float(ret.charges)
        return ret

    def to_raw(self, copy: bool=False, check: bool = True):
        ret = self if not copy else self.copy()
        if not ret._is_raw:
            if check:
                raise Exception("call to_foramt() on formatted Info object")
            return ret
        ret.age = str(age)
        ret.sex = "male" if ret.sex else "female"   # True 男, False 女
        ret.bmi = str(ret.bmi)
        ret.children = str(ret.children)
        ret.smoker = 'yes' if ret.smoker else 'no'
        ret.region = Regions_to_str[ret.region]
        ret.charges = str(ret.charges)
        return ret

    def to_vector(self)->(list, float):
        origin =  self if not self.is_raw() else self.to_format(copy=True, check=False)
        # age, sex, bmi, children, smoker, se, sw, ne, nw
        #                                 |--  one-hot  --|
        ret = [0] * 9
        ret[0] = origin.age
        ret[1] = 1 if origin.sex else 0
        ret[2] = origin.bmi
        ret[3] = origin.children
        ret[4] = 1 if origin.smoker else 0
        ret[5 + int(origin.region)] = 1
        return ret, origin.charges

class Loader:
    def __init__(self, csv_file_name: str):
        self.csv_file = open(csv_file_name, 'r', encoding='utf8')
        self.csv_file.readline()        # remove header

    def __iter__(self):
        return self

    def __next__(self):
        ret = self.csv_file.readline().strip()
        if not ret:
            raise StopIteration
        age, sex, bmi, children, smoker, region, charges = tuple(
            ret.split(','))
        return Info(age, sex, bmi, children, smoker, region, charges, True)

def origin_walk_csv(csv_file_name) -> Loader:
    return Loader(csv_file_name)

def origin_list_csv(csv_file_name) -> list:
    return list(origin_walk_csv(csv_file_name))

def format_walk_csv(csv_file_name) -> "Iterable[Info]":
    return map(lambda info: info.to_foramt, origin_walk_csv(csv_file_name))


def format_list_csv(csv_file_name) -> list:
    return list(format_walk_csv(csv_file_name))

def vector_walk_csv(csv_file_name) -> "Iterable[tuple[list, float]]":
    return map(lambda info: info.to_vector(), origin_walk_csv(csv_file_name))

def vector_list_csv(csv_file_name) -> list:
    return list(vector_walk_csv(csv_file_name))

def load_all_csv_as_ndarray(csv_file_name) -> (np.ndarray, np.ndarray):
    sample_count = sum(1 for _ in origin_walk_csv(csv_file_name))
    samples = np.empty((sample_count, 9))
    labels = np.empty((sample_count, 1))
    for row, (sample, label) in enumerate(vector_walk_csv(csv_file_name)):
        samples[row] = sample
        labels[row][0] = label
    return samples, labels

if __name__ == "__main__":
    for vector, charges in vector_list_csv(config.PATH.DATA_TRAIN_CSV):
        print(vector, charges)
        assert sum(vector[5:9]) == 1
    test_samples, test_labels = load_all_csv_as_ndarray(config.PATH.DATA_TEST_CSV)
    train_samples, train_labels = load_all_csv_as_ndarray(config.PATH.DATA_TRAIN_CSV)