'''
Created on 12.05.2019

@author: Philipp
'''
import numpy as np


def __normalize_strings(y_dataset):
    """
        Convert byte strings to normal strings.
    """
    return [label.decode() for label in y_dataset if not isinstance(label, str)]


def convert_label_to_ids(y_dataset, label_to_id):
    """
        Convert a y_dataset that is based on label names to corresponding id numbers.
        For example: (deer, tree, human) becomes (1, 0, 0) for {b'tree': 0, b'human':0, b'deer':1}
    """
    y_dataset = np.array(__normalize_strings(y_dataset))
    for label in label_to_id:
        label_idx = np.squeeze(np.argwhere(y_dataset == label))
        
        if np.shape(label_idx)[0] != 0:
            y_dataset[label_idx] = label_to_id[label]
    return y_dataset.astype(np.uint32)