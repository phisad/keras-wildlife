'''
Created on 13.05.2019

@author: Philipp
'''
from wildlife.dataset.imagenet import list_imagenet
from wildlife.dataset.images.mappings import list_sorted, mappings_to_tuples, \
    filter_large_mapping
import numpy as np
from wildlife.dataset import write_csv_splits
from wildlife.dataset.images.categories import split_categories


def create_imagenet_dataset_splits(source_directory, target_split_directory_name):
    if not source_directory.endswith("/"):
        source_directory = source_directory + "/"
        
    mappings = list_imagenet(source_directory)
    
    large_mappings = filter_large_mapping(mappings)
    
    list_sorted(large_mappings)
    
    large_mappings = mappings_to_tuples(large_mappings)

    # We need to create train and train_dev on the same run, so that train images are not within the train_dev set
    categories = [
        (large_mappings["hare"],       400, [300, 100]),
        (large_mappings["marten"],     400, [300, 100]),
        (large_mappings["wildboar"],   800, [700, 100]),
        (large_mappings["dog"],       1000, [900, 100]),
        (large_mappings["cat"],       1000, [900, 100]),
        (large_mappings["horse"],     1000, [900, 100]),
        (large_mappings["deer"],      1000, [900, 100]),
        (large_mappings["hedgehog"],  1000, [900, 100]),
        (large_mappings["squirrel"],  1000, [900, 100]),
        (large_mappings["bird"],      1000, [900, 100]),
        (large_mappings["car"],       1000, [900, 100]),
        (large_mappings["tractor"],   1000, [900, 100]),
        (large_mappings["tree"],      1000, [900, 100]),
        (large_mappings["fox"],       1000, [900, 100]),
        (large_mappings["humankind"], 1000, [900, 100]),
        (large_mappings["racoon"],    1000, [900, 100])
    ]
    
    count_arr = np.array([listing for _,_, listing in categories])
    print("Total               : {}".format(np.sum(np.sum(count_arr, axis=0))))
    
    count_total_per_split = np.sum(count_arr, axis=0)
    print("Total per split     : {}".format(count_total_per_split))
    
    count_animal = np.array([listing for tups,_, listing in categories if tups[0][1] not in ["tree", "humankind", "car", "tractor"]])
    count_animal_per_split = np.sum(count_animal, axis=0)
    print("Animals per split   : {}".format(count_animal_per_split))
    
    count_bg_per_split = count_total_per_split - count_animal_per_split
    print("Background per split: {} ({})".format(count_bg_per_split, np.around(count_bg_per_split / count_total_per_split, 2)))
    
    idx_to_label = [tups[0][1] for tups,_, listing in categories]
    idx_to_label = dict([(idx, label) for idx, label in enumerate(idx_to_label)])
    print("Index to label      : \n{}".format(idx_to_label))
    print()
    
    for split_idx, reference_value in [(0, 900), (1, 100)]:
        class_weights = np.around(1.0 / (count_arr[:,split_idx] / reference_value), 1)
        class_weights_dict = dict([(idx_to_label[idx], w) for idx, w in enumerate(class_weights)])
        print("Class weights       : \n{}".format(class_weights_dict))
        print()
        
    splits = split_categories(categories)
    for split in splits:
        print(len(split))
        
    write_csv_splits(splits, 
                     filenames=["source_train.csv","source_dev.csv"], 
                     directory=source_directory + target_split_directory_name)
