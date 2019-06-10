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
        if np.ndim(label_idx) == 0:
            # scalar to list array
            label_idx = np.array([label_idx])
        first_dimension = np.shape(label_idx)[0]
        if first_dimension > 0: 
            # put label everywhere in y_dataset
            y_dataset[label_idx] = label_to_id[label]
    y_dataset = y_dataset.astype(np.uint32) 
    return y_dataset
