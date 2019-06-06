'''
Created on 11.05.2019

@author: Philipp
'''
from os import listdir
from os.path import join
import numpy as np
from wildlife.dataset.images import read_single_rgb


def filter_large_mapping(mapping):
    large_mapping = {}
    for cls in mapping:
        large_mapping[cls] = __filter_large_listing(mapping[cls])
    return large_mapping


def __filter_large_listing(listing):
    large_listing = []
    total = len(listing)
    counter = 0
    for image_file in listing:
        counter += 1
        print('>> Checking image %d/%d' % (counter, total), end="\r")
        with read_single_rgb(image_file) as image:
            if np.shape(image) >= (224, 224, 3):
                large_listing.append(image_file)
    print()
    return large_listing


def mappings_to_tuples(mappings, prefix="", label_renaming={}):
    """
        Convert mapping of "cls" -> [filepath] to "cls" -> [(filepath, cls)]
    """
    tuple_mappings = {}
    for label in mappings:
        label_mappings = mappings[label]
        if label in label_renaming:
            label = label_renaming[label]
        tuples = [(prefix + "/" + filepath, label) for filepath in label_mappings]
        if label in tuple_mappings:
            tuple_mappings[label].extend(tuples)
        else:
            tuple_mappings[label] = tuples
    return tuple_mappings


def list_sorted(mapping):
    from collections import OrderedDict
    for animal, images in OrderedDict(sorted(mapping.items(), key=lambda t: len(t[1]))).items():
        print("{:15}: {:6}".format(animal, len(images)))
    print("{:15}: {:6}".format("Total", __count_total(mapping)))
    print()


def __count_total(mapping):
    return sum([len(values) for _, values in mapping.items()])


def __list_directory(directory):
    return [join(directory, file) for file in listdir(directory)]
