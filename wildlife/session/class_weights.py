'''
Created on 12.05.2019

@author: Philipp
'''
from sklearn.utils import class_weight as skweights
import numpy as np


def __list_sorted(weight_mappings, title_mappings=None):
    from collections import OrderedDict
    for cls, weight in OrderedDict(sorted(weight_mappings.items(), key=lambda t: t[1])).items():
        if title_mappings:
            cls = title_mappings[cls]
        print("{:10}: {:6}".format(cls, weight))
    print("{:10}: {:6}".format("Total", np.around(__count_total(weight_mappings), 1)))
    print()

    
def __count_total(weight_mappings):
    return sum([values for _, values in weight_mappings.items()])


def calculate_class_weights(y_dataset, title_mappings=None):
    """
        Use the scikit-lean weighting algorithm to balance the weights for the unbalanced animal categories.
        
        @param y_dataset: list
            The animal categories labelled as integers e.g. [0 1 0 2 3]
    """
    print("Calculate class weights: balanced")
    class_weights = skweights.compute_class_weight('balanced', np.unique(y_dataset), y_dataset)
    class_weights = dict([(cls_idx, np.around(w, 1)) for cls_idx, w in enumerate(class_weights)])
    __list_sorted(class_weights, title_mappings)
    return class_weights
