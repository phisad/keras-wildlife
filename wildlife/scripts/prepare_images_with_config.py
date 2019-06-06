#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser
from wildlife.configuration import Configuration
from wildlife.dataset.wildlife.splits import create_wildlife_dataset_splits
from wildlife.dataset.imagenet.splits import create_imagenet_dataset_splits
from wildlife.dataset.images import get_preprocessing_tfrecord_file
from wildlife.dataset.images.tfrecords import create_tfrecords_by_csv_from_config


def to_split_dir(directory_path, split_name):
    return "/".join([directory_path, split_name])


def main():
    parser = ArgumentParser("Prepare the dataset for training")
    parser.add_argument("command", help="""One of [csv, preprocess, all].
                        csv: Write the dataset splits into csv files
                        preprocess: Resizes images and stores them by image id in a TFRecord file 
                        all: All of the above""")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-d", "--dataset", default="wildlife", help="The dataset to operate on. One of [wildlife, imagenet]. Default: wildlife")
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
        
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    
    if not run_opts.dataset and not run_opts.dataset in ["wildlife", "imagenet"]:
        raise Exception("Cannot prepare, when dataset is not chosen. Please provide the dataset using the '-d' option and retry.")
        
    print("Starting image preparation: {}".format(run_opts.command))
    
    target_dir = config.getDatasetDirectoryPath()
    
    if run_opts.command in ["all", "csv"]:
        if run_opts.dataset == "wildlife":
            print("csv: Write the dataset splits into csv files for 'wildlife' using " + target_dir)
            dataset_dir = config.getWildlifeDatasetDirectoryPath()
            create_wildlife_dataset_splits(dataset_dir, target_dir, "wl-c11")
        
        if run_opts.dataset == "imagenet":
            print("csv: Write the dataset splits into csv files for 'imagenet' using " + target_dir)
            dataset_dir = config.getImagenetDatasetDirectoryPath()
            create_imagenet_dataset_splits(dataset_dir, target_dir, "in-c16")
    
    if run_opts.command in ["all", "preprocess"]:
        if run_opts.dataset == "wildlife":
            print("preprocess: Resizes images and stores them by image id in a TFRecord file for 'wildlife' using " + target_dir)
            dataset_dir = config.getWildlifeDatasetDirectoryPath()
            split_name = "wl-c11"
            split_files = ["target_train", "target_dev", "target_test"]
        
        if run_opts.dataset == "imagenet":
            print("preprocess: Resizes images and stores them by image id in a TFRecord files for 'imagenet' using " + target_dir)
            dataset_dir = config.getImagenetDatasetDirectoryPath()
            split_name = "in-c16"
            split_files = ["source_train", "source_dev"]
            
        tfrecord_file = get_preprocessing_tfrecord_file(dataset_dir, split_name)
        if tfrecord_file:
            print("Skip preprocessing for split '{}' because TFRecord file already exists at {}".format(split_name, tfrecord_file))
        else:
            create_tfrecords_by_csv_from_config(config, to_split_dir(dataset_dir, split_name), split_files)


if __name__ == '__main__':
    main()
    
