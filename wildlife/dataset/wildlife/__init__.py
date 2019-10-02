'''
Created on 11.05.2019

@author: Philipp
'''

from os import listdir
from os.path import isdir
from os.path import join
from os.path import basename
import collections
from collections import OrderedDict
import numpy as np
import csv
from wildlife.dataset.images.mappings import list_sorted
    
ANIMAL_LABELS = ["Rabbit", "Hedgehog", "Marten", "Racoon", "Cat", "Wild_Boar",
                 "Fox", "Squirrel", "Horse", "Vole", "Roe_deer" , "Hare", "Fallow_deer", "Dog", "Bird"]

"""
    The initial dataset returned:
        Hedgehog       :      3
        Rabbit         :      3
        Squirrel       :      9
        Vole           :     15
        Horse          :     66
        Wild_Boar      :    132
        Racoon         :    171
        Marten         :    213
        Fox            :    327
        Cat            :    375
        Dog            :    433
        Fallow_deer    :   1029
        Hare           :   1146
        Bird           :   1512
        Roe_deer       :   3192
        background     : 129810
        Total          : 138436
"""
def list_wildlife_labelled(labelfile="/data/wildlife/label.csv", select_classes=[]):
    animals = ANIMAL_LABELS
    if select_classes:
        animals = [animal for animal in animals if animal in select_classes]
    
    mappings = collections.defaultdict(list)
    duplicates = 0
    with open(labelfile) as f:
        reader = csv.reader(f)
        for line in reader:
            line_file = line[0]
            line_labels = line[1:]
            
            animal_labels = []
            for animal_label in animals:
                if animal_label in line_labels:
                    animal_labels.append(animal_label)       

            if len(animal_labels) > 1:
                print("Duplicates: " + str(animal_labels))
                duplicates += 1
                continue

            if len(animal_labels) == 1:
                for animal_label in animal_labels:
                    mappings[animal_label].append(line_file)

            if not animal_labels and "none" in line_labels:
                if not select_classes:
                    mappings["background"].append(line_file)
                    continue
                    
                if "background" in select_classes:
                    mappings["background"].append(line_file)
                
    list_sorted(mappings)
    return mappings, duplicates
