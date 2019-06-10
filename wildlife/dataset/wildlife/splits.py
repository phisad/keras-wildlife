'''
Created on 13.05.2019

@author: Philipp
'''
import numpy as np
from wildlife.dataset import write_csv_splits
from wildlife.dataset.wildlife import list_wildlife_labelled
from wildlife.dataset.images.categories import split_categories
from wildlife.dataset.images.mappings import mappings_to_tuples


def create_wildlife_dataset_splits(source_directory, target_directory, target_split_name, method="weighted"):
    """
        @param method: str
            One of [weighted, single, small]
    """
    mappings, _ = list_wildlife_labelled(labelfile=source_directory + "/label.csv")
    
    # label merging (but here rather to align to source dataset)
    # natural merge of deer
    label_renaming = {
        "Roe_deer"    : "deer",  # natural label merge
        "Fallow_deer" : "deer",  # natural label merge
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
    """ this could also be made generally configurable """
    if method == "single":
        __create_wildlife_single_split_by_categories(tuple_mappings, target_directory, target_split_name)
    if method == "weighted":
        __create_wildlife_weighted_splits_by_categories(tuple_mappings, target_directory, target_split_name)
    if method == "small":
        __create_wildlife_small_splits_by_categories(tuple_mappings, target_directory, target_split_name)


def __create_wildlife_single_split_by_categories(tuple_mappings, target_directory, target_split_name):    
    # We need to create train and train_dev on the same run, so that train images are not within the train_dev set
    categories = [
        (tuple_mappings["horse"], 66, [ 66]),
        (tuple_mappings["wildboar"], 132, [132]),
        (tuple_mappings["racoon"], 171, [171]),
        (tuple_mappings["marten"], 213, [213]),
        (tuple_mappings["fox"], 327, [327]),
        (tuple_mappings["cat"], 375, [375]),
        (tuple_mappings["dog"], 433, [433]),
        (tuple_mappings["hare"], 1146, [1146]),
        (tuple_mappings["bird"], 1512, [1512]),
        (tuple_mappings["deer"], 4221, [4221]),
        (tuple_mappings["background"], 129810, [129810]),
    ]
    count_arr = np.array([listing for _, _, listing in categories])
    print("Total               : {}".format(np.sum(np.sum(count_arr, axis=0))))
    
    count_total_per_split = np.sum(count_arr, axis=0)
    print("Total per split     : {}".format(count_total_per_split))
    
    count_animal = np.array([listing for tups, _, listing in categories if tups[0][1] != "background"])
    count_animal_per_split = np.sum(count_animal, axis=0)
    print("Animals per split   : {}".format(count_animal_per_split))
    
    count_bg_per_split = count_total_per_split - count_animal_per_split
    print("Background per split: {} ({})".format(count_bg_per_split, np.around(count_bg_per_split / count_total_per_split, 2)))
    print()
        
    splits = split_categories(categories)
        
    write_csv_splits(splits,
                     filenames=["target_all.csv"],
                     directory=target_directory + "/" + target_split_name)


def __create_wildlife_weighted_splits_by_categories(tuple_mappings, target_directory, target_split_name):    
    # We need to create train and train_dev on the same run, so that train images are not within the train_dev set
    categories = [
        (tuple_mappings["horse"], 60, [  40, 10, 10]),
        (tuple_mappings["wildboar"], 130, [  90, 20, 20]),
        (tuple_mappings["racoon"], 170, [ 120, 25, 25]),
        (tuple_mappings["marten"], 210, [ 150, 30, 30]),
        (tuple_mappings["fox"], 320, [ 230, 45, 45]),
        (tuple_mappings["cat"], 370, [ 260, 55, 55]),
        (tuple_mappings["dog"], 430, [ 300, 65, 65]),
        (tuple_mappings["hare"], 1140, [ 800, 165, 165]),
        (tuple_mappings["bird"], 1510, [1050, 230, 230]),
        (tuple_mappings["deer"], 4210, [3000, 605, 605]),
        (tuple_mappings["background"], 8540, [6040, 1250, 1250]),
    ]
    count_arr = np.array([listing for _, _, listing in categories])
    print("Total               : {}".format(np.sum(np.sum(count_arr, axis=0))))
    
    count_total_per_split = np.sum(count_arr, axis=0)
    print("Total per split     : {}".format(count_total_per_split))
    
    count_animal = np.array([listing for tups, _, listing in categories if tups[0][1] != "background"])
    count_animal_per_split = np.sum(count_animal, axis=0)
    print("Animals per split   : {}".format(count_animal_per_split))
    
    count_bg_per_split = count_total_per_split - count_animal_per_split
    print("Background per split: {} ({})".format(count_bg_per_split, np.around(count_bg_per_split / count_total_per_split, 2)))
    print()
        
    splits = split_categories(categories)
        
    write_csv_splits(splits,
                     filenames=["target_train.csv", "target_dev.csv", "target_test.csv"],
                     directory=target_directory + "/" + target_split_name)


def __create_wildlife_small_splits_by_categories(tuple_mappings, target_directory, target_split_name):    
    # We need to create train and train_dev on the same run, so that train images are not within the train_dev set
    categories = [
        (tuple_mappings["horse"],         60, [ 10, 25, 25]),
        (tuple_mappings["wildboar"],     130, [ 10, 60, 60]),
        (tuple_mappings["racoon"],       170, [ 10, 80, 80]),
        (tuple_mappings["marten"],       210, [ 10, 100, 100]),
        (tuple_mappings["fox"],          320, [ 20, 150, 150]),
        (tuple_mappings["cat"],          370, [ 20, 175, 175]),
        (tuple_mappings["dog"],          430, [ 30, 200, 200]),
        (tuple_mappings["hare"],        1140, [ 80, 530, 530]),
        (tuple_mappings["bird"],        1510, [100, 705, 705]),
        (tuple_mappings["deer"],        4210, [300, 1955, 1955]),
        (tuple_mappings["background"],  8540, [600, 3970, 3970]),
    ]
    return __create_splits_by_categories(categories, target_directory, target_split_name, target_split_files=["target_train.csv", "target_dev.csv", "target_test.csv"])

    
def __create_splits_by_categories(categories, target_directory, target_split_name, target_split_files):
    count_arr = np.array([listing for _, _, listing in categories])
    print("Total               : {}".format(np.sum(np.sum(count_arr, axis=0))))
    
    count_total_per_split = np.sum(count_arr, axis=0)
    print("Total per split     : {}".format(count_total_per_split))
    
    count_animal = np.array([listing for tups, _, listing in categories if tups[0][1] != "background"])
    count_animal_per_split = np.sum(count_animal, axis=0)
    print("Animals per split   : {}".format(count_animal_per_split))
    
    count_bg_per_split = count_total_per_split - count_animal_per_split
    print("Background per split: {} ({})".format(count_bg_per_split, np.around(count_bg_per_split / count_total_per_split, 2)))
    print()
        
    splits = split_categories(categories)
        
    write_csv_splits(splits,
                     filenames=target_split_files,
                     directory=target_directory + "/" + target_split_name)
