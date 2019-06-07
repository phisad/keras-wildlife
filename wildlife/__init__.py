import numpy as np


def is_multiclass(y_train_cat):
    return get_dimensions(y_train_cat) > 2

    
def get_dimensions(y_train_cat):
    return np.shape(y_train_cat)[1]


def to_split_dir(directory_path, split_name):
    return "/".join([directory_path, split_name])
