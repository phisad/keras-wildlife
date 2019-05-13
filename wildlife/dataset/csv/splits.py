'''
Created on 03.05.2019

@author: Philipp
'''
import os

import csv

                
def __to_list(string):
    string = string.replace("\"", "")
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace("'", "")
    return [s.strip() for s in string.split(",")]


def __to_unix(string):
    return string.replace("\\", "/")

            
def write_csv_splits(splits, filenames=["train.csv", "train_dev.csv"], mode="w", directory="."):
    if len(splits) != len(filenames):
        raise Exception("Please hand in as many filenames as splits")
        
    if not os.path.exists(directory):
        os.mkdir(directory)
        
    for idx, filename in enumerate(filenames):
        with open("/".join([directory, filename]), mode) as f:
            writer = csv.DictWriter(f, ["filepath", "labels"])
            if mode == "w":
                writer.writeheader()
            total = len(splits[idx])
            counter = 0
            for filepath, labels in splits[idx]:
                counter += 1
                print('>> Write image tuple to %s (%d/%d)' % (filename, counter, total), end="\r")
                writer.writerow({"filepath":filepath, "labels": labels})
            print()


def read_csv(csv_file_path):
    image_files = []
    with open(csv_file_path) as f:
        reader = csv.DictReader(f)
        for line in reader:
            image_files.append((__to_unix(line["filepath"]), __to_list(line["labels"])))
    return image_files
