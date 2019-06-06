'''
Created on 13.05.2019

@author: Philipp
'''
import numpy as np
from wildlife.dataset import write_csv_splits
from wildlife.dataset.wildlife import list_wildlife_labelled
from wildlife.dataset.images.categories import split_categories
from wildlife.dataset.images.mappings import mappings_to_tuples, list_sorted


def create_wildlife_dataset_splits(source_directory, target_split_directory_name):
    if not source_directory.endswith("/"):
        source_directory = source_directory + "/"
        
    mappings, _ = list_wildlife_labelled(labelfile=source_directory + "label.csv")
    
    # label merging (but here rather to align to source dataset)
    # natural merge of deer
    label_renaming = {
        "Roe_deer"    : "deer", # natural label merge
        "Fallow_deer" : "deer", # natural label merge
        "Marten"      : "marten", 
        "Hare"        : "hare", 
        "Bird"        : "bird", 
        "Squirrel"    : "squirrel", 
        "Wild_Boar"   : "wildboar", 
        "Hedgehog"    : "hedgehog", 
        "Racoon"      : "racoon", 
        "Fox"         : "fox",
        "Horse"       : "horse",
        "Dog"         : "dog",
        "Cat"         : "cat"
    }
    
    tuple_mappings = mappings_to_tuples(mappings, prefix=source_directory, label_renaming=label_renaming)
    list_sorted(tuple_mappings)
    
    # We need to create train and train_dev on the same run, so that train images are not within the train_dev set
    categories = [
        (tuple_mappings["horse"],              60,  [  40,   10,   10]),
        (tuple_mappings["wildboar"],          130,  [  90,   20,   20]),
        (tuple_mappings["racoon"],            170,  [ 120,   25,   25]),
        (tuple_mappings["marten"],            210,  [ 150,   30,   30]),
        (tuple_mappings["fox"],               320,  [ 230,   45,   45]),
        (tuple_mappings["cat"],               370,  [ 260,   55,   55]),
        (tuple_mappings["dog"],               430,  [ 300,   65,   65]),
        (tuple_mappings["hare"],              1140, [ 800,  165,  165]),
        (tuple_mappings["bird"],              1510, [1050,  230,  230]),
        (tuple_mappings["deer"],              4210, [3000,  605,  605]),
        (tuple_mappings["background"],        8540, [6040, 1250, 1250]),
    ]
    count_arr = np.array([listing for _,_, listing in categories])
    print("Total               : {}".format(np.sum(np.sum(count_arr, axis=0))))
    
    count_total_per_split = np.sum(count_arr, axis=0)
    print("Total per split     : {}".format(count_total_per_split))
    
    count_animal = np.array([listing for tups,_, listing in categories if tups[0][1] != "background"])
    count_animal_per_split = np.sum(count_animal, axis=0)
    print("Animals per split   : {}".format(count_animal_per_split))
    
    count_bg_per_split = count_total_per_split - count_animal_per_split
    print("Background per split: {} ({})".format(count_bg_per_split, np.around(count_bg_per_split / count_total_per_split, 2)))
    print()
    
    idx_to_label = [tups[0][1] for tups,_, listing in categories]
    idx_to_label = dict([(idx, label) for idx, label in enumerate(idx_to_label)])
    print("Index to label      : \n{}".format(idx_to_label))
    print()
    
    for split_idx, reference_value in [(0, 6030), (1, 1255), (2, 1255)]:
        print("Class weights for Split / Reference: {} / {}".format(split_idx, reference_value))
        class_weights = np.around(1.0 / (count_arr[:,split_idx] / reference_value), 0)
        class_weights_dict = dict([(idx_to_label[idx], w) for idx, w in enumerate(class_weights)])
        print("{}".format(class_weights_dict))
        print()
        
    splits = split_categories(categories)
    for split in splits:
        print(len(split))
        
    write_csv_splits(splits,
                     filenames=["target_train.csv", "target_dev.csv", "target_test.csv"],
                     directory=source_directory + target_split_directory_name)
