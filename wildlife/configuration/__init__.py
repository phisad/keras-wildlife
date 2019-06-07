'''
Created on 07.03.2019

@author: Philipp
'''
import configparser
import os.path as path
import os
import numpy as np

import json
from wildlife.scripts import OPTION_DRY_RUN
SECTION_DATASET = "DATASETS"
OPTION_WILDLIFE_DATASET_DIRECTORY_PATH = "WildlifeDatasetDirectoryPath"
OPTION_IMAGENET_DATASET_DIRECTORY_PATH = "ImagenetDatasetDirectoryPath"
OPTION_DATASET_DIRECTORY_PATH = "DatasetDirectoryPath"

SECTION_MODEL = "MODEL"
OPTION_PRINT_MODEL_SUMMARY = "PrintModelSummary"
OPTION_MODEL_TYPE = "ModelType"
OPTION_MODEL_CLASSIFIER = "ModelClassifier"
OPTION_USE_BATCH_NORMALIZATION = "UseBatchNormalization"
OPTION_IMAGE_INPUT_SHAPE = "ImageInputShape"

SECTION_TRAINING = "TRAINING"
OPTION_GPU_DEVICES = "GpuDevices"
OPTION_TENSORBOARD_LOGGING_DIRECTORY = "TensorboardLoggingDirectory"
OPTION_EPOCHS = "Epochs"
OPTION_BATCH_SIZE = "BatchSize"
OPTION_USE_MULTI_PROCESSING = "UseMultiProcessing"
OPTION_WORKERS = "Workers"
OPTION_MAX_QUEUE_SIZE = "MaxQueueSize"

FILE_NAME = "configuration.ini"


def store_configuration(configuration, target_directory_path_or_file, split_name):
    lookup_filename = FILE_NAME
    if split_name:    
        lookup_filename = "configuration_{}.ini".format(split_name) 
    return store_config_to(configuration.config, target_directory_path_or_file, lookup_filename)


def store_config_to(config, directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename, to_read=False)
    print("Persisting configuration to " + file_path)    
    with open(file_path, "w") as config_file:
        config.write(config_file)
    return file_path


def determine_file_path(directory_or_file, lookup_filename, to_read=True):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = directory_or_file
    if os.path.isdir(directory_or_file):
        if lookup_filename == None:
            raise Exception("Cannot determine source file in directory without lookup_filename")
        file_path = "/".join([directory_or_file, lookup_filename])
    if to_read and not os.path.isfile(file_path):
        raise Exception("There is no such file in the directory to read: " + file_path)
    return file_path


class Configuration(object):

    def __init__(self, config_path=None):
        '''
        Constructor
        '''
        self.run_opts = {}
        self.config = configparser.ConfigParser()
        if not config_path:
            config_path = Configuration.config_path()
        print("Use configuration file at: " + config_path)
        self.config.read(config_path)
        
    def __getitem__(self, idx):
        return self.run_opts[idx]
    
    def __setitem__(self, key, value):
        self.run_opts[key] = value

    def is_dryrun(self):
        if self[OPTION_DRY_RUN]:
            return self[OPTION_DRY_RUN]
        return False

    def getPrintModelSummary(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_PRINT_MODEL_SUMMARY)
    
    def getUseBatchNormalization(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_USE_BATCH_NORMALIZATION)
    
    def getModelType(self):
        return self.config.get(SECTION_MODEL, OPTION_MODEL_TYPE)
    
    def getModelClassifier(self):
        return self.config.get(SECTION_MODEL, OPTION_MODEL_CLASSIFIER)
    
    def getWildlifeDatasetDirectoryPath(self):
        return self.config.get(SECTION_DATASET, OPTION_WILDLIFE_DATASET_DIRECTORY_PATH)
    
    def getImagenetDatasetDirectoryPath(self):
        return self.config.get(SECTION_DATASET, OPTION_IMAGENET_DATASET_DIRECTORY_PATH)
    
    def getDatasetDirectoryPath(self):
        return self.config.get(SECTION_DATASET, OPTION_DATASET_DIRECTORY_PATH)
    
    def getImageInputShape(self):
        shape = self.config.get(SECTION_MODEL, OPTION_IMAGE_INPUT_SHAPE)
        shape_tuple = tuple(map(int, shape.strip('()').split(',')))
        return shape_tuple

    def getGpuDevices(self):
        return self.config.getint(SECTION_TRAINING, OPTION_GPU_DEVICES)
    
    def getTensorboardLoggingDirectory(self):
        return self.config.get(SECTION_TRAINING, OPTION_TENSORBOARD_LOGGING_DIRECTORY)

    def getEpochs(self):
        return self.config.getint(SECTION_TRAINING, OPTION_EPOCHS)

    def getBatchSize(self):
        return self.config.getint(SECTION_TRAINING, OPTION_BATCH_SIZE)
    
    def getUseMultiProcessing(self):
        return self.config.getboolean(SECTION_TRAINING, OPTION_USE_MULTI_PROCESSING)    

    def getWorkers(self):
        return self.config.getint(SECTION_TRAINING, OPTION_WORKERS)    

    def getMaxQueueSize(self):
        return self.config.getint(SECTION_TRAINING, OPTION_MAX_QUEUE_SIZE)    
    
    def dump(self):
        print("Configuration:")
        for section in self.config.sections():
            print("[{}]".format(section))
            for key in self.config[section]:
                print("{} = {}".format(key, self.config[section][key]))
                
    @staticmethod
    def config_path():
        # Lookup file in project root or install root
        project_root = os.path.dirname(os.path.realpath(__file__))
        config_path = "/".join([project_root, FILE_NAME])
        if path.exists(config_path):
            return config_path
        print("Warn: No existing 'configuration.ini' at default location " + config_path)
        
        # Lookup file in user directory
        from pathlib import Path
        home_directory = str(Path.home())
        config_path = "/".join([home_directory, "wildlife-" + FILE_NAME])
        if path.exists(config_path):
            return config_path
        print("Warn: No existing 'wildlife-configuration.ini' file at user home " + config_path)
        
        raise Exception("""Please place a 'configuration.ini' in the default location 
                            or a 'wildlife-configuration.ini' in your home directory 
                            or use the run option to specify a specific file""")

