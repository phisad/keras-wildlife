SPLIT_TRAIN = "train"
SPLIT_VALIDATE = "validate"
SPLIT_TEST = "test"
SPLIT_TEST_DEV = "test_dev"
SPLIT_TRAINVAL = "trainval"

import numpy as np


def __is_multiclass(y_train_cat):
    return __get_dimensions(y_train_cat) > 2

    
def __get_dimensions(y_train_cat):
    return np.shape(y_train_cat)[1]

def to_split_dir(directory_path, split_name):
    return "/".join([directory_path, split_name])
