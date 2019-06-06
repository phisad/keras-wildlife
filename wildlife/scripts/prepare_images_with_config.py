#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser
from wildlife.configuration import Configuration
from wildlife.scripts import OPTION_DRY_RUN
from wildlife.dataset.wildlife.splits import create_wildlife_dataset_splits
from wildlife.dataset.imagenet.splits import create_imagenet_dataset_splits


def main():
    parser = ArgumentParser("Prepare the dataset for training")
    parser.add_argument("command", help="""One of [csv, preprocess, all].
                        csv: Write the dataset splits into csv files
                        preprocess: Resizes images and stores them by image id in a TFRecord file 
                        all: All of the above""")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-d", "--dataset", default="wildlife", help="The dataset to operate on. One of [wildlife, imagenet]. Default: wildlife")
    parser.add_argument("-b", "--batch_size", type=int)
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
        
    config[OPTION_DRY_RUN] = run_opts.dryrun
    config["batch_size"] = run_opts.batch_size
    config["num_images"] = run_opts.num_images

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    
    if not run_opts.dataset and not run_opts.dataset in ["wildlife", "imagenet"]:
        raise Exception("Cannot prepare, when dataset is not chosen. Please provide the dataset using the '-d' option and retry.")
        
    print("Starting image preparation: {}".format(run_opts.command))
    
    if run_opts.command in ["all", "csv"]:
        print("csv: Write the dataset splits into csv files")
        if run_opts.dataset == "wildlife":
            dataset_dir = config.getWildlifeDatasetDirectoryPath()
            create_wildlife_dataset_splits(dataset_dir, "wl-c11")
        
        if run_opts.dataset == "imagenet":
            dataset_dir = config.getImagenetDatasetDirectoryPath()
            create_imagenet_dataset_splits(dataset_dir, "in-c16")
    
    """         
    if run_opts.command in ["all", "preprocess"]:
        print("preprocess: Resizes images and stores them by image id in a TFRecord file")
        for split_name in split_names:
            tfrecord_file = get_preprocessing_tfrecord_file(directory_path, split_name)
            if tfrecord_file:
                print("Skip preprocessing for split '{}' because TFRecord file already exists at {}".format(split_name, tfrecord_file))
            else:
                target_shape = config.getImageInputShape()
                image_paths = _get_image_paths(to_split_dir(directory_path, split_name))
                preprocess_images_and_write_tfrecord(image_paths, directory_path, target_shape, split_name)
    """


if __name__ == '__main__':
    main()
    
