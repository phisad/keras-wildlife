#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser
from wildlife.configuration import Configuration
from wildlife.scripts import OPTION_DRY_RUN
from wildlife.training.baseline import start_training_baseline_from_config


def main():
    parser = ArgumentParser("Start the model using the configuration.ini")
    parser.add_argument("command", help="""One of [training, predict]. 
                        training: Start training with the configuration.
                        predict: Apply a model on the dataset with the configuration and write the result file""")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-t", "--model_type", help="The model type. One of [baseline].")
    parser.add_argument("-f", "--path_to_model", help="The absolute path to the model to predict or continue training.")
    parser.add_argument("-i", "--initial_epoch", type=int, help="The initial epoch to use when continuing training. This is required for continuing training.")
    parser.add_argument("-s", "--split_name", help="""The split name to perform the prediction or training on. This is required for predict. For example 'wl-c11'.""")
    parser.add_argument("-d", "--dryrun", action="store_true")
    parser.add_argument("-m", "--do_multiclass", action="store_true", default=False)
    
    run_opts = parser.parse_args()
    
    if not run_opts.split_name:
        raise Exception("Please the split name and retry.")
    split_name = run_opts.split_name
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
    config[OPTION_DRY_RUN] = run_opts.dryrun
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    
    config.dump()
    
    dataset_dir = config.getDatasetDirectoryPath()
    if run_opts.command == "training":
        if not run_opts.model_type:
            raise Exception("Please provide the model type and retry.")
        
        if run_opts.model_type == "baseline":
            start_training_baseline_from_config(config, dataset_dir, split_name, do_multiclass=run_opts.do_multiclass)
    
        
if __name__ == '__main__':
    main()
    
