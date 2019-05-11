'''
Created on 11.05.2019

@author: Philipp
'''

from os import listdir
from os.path import join
from wildlife.dataset.wildlife.mappings import list_sorted


def list_imagenet(directory="/data/wildlife-project/imagenet", select_classes=[]):
    classes = listdir(directory)
    mappings = {}
    for cls in classes:
        if select_classes and not cls in select_classes:
            continue
        mappings[cls] = __list_directory(join(directory, cls))
    list_sorted(mappings)
    return mappings


def __list_directory(directory):
    return [join(directory, file) for file in listdir(directory)]
