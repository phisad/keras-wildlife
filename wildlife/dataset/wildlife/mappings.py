'''
Created on 11.05.2019

@author: Philipp
'''
from os import listdir
from os.path import join


def mappings_to_tuples(mappings, prefix="", label_renaming={}):
    """
        Convert mapping of "cls" -> [filepath] to "cls" -> [(filepath, cls)]
    """
    tuple_mappings = {}
    for label in mappings:
        label_mappings = mappings[label]
        if label in label_renaming:
            label = label_renaming[label]
        tuples = [(prefix + filepath, label) for filepath in label_mappings]
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
