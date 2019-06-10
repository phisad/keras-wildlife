#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser
from wildlife.configuration import Configuration
from wildlife.scripts import OPTION_DRY_RUN
from wildlife.session.baseline import start_evaluate_baseline_in_memory, \
    start_training_baseline_from_config, start_evaluate_baseline


def main():
    parser = ArgumentParser("Start the model using the configuration.ini")
    parser.add_argument("command", help="""One of [training, evaluate]. 
                        training: Start training with the configuration.
                        evaluate: Apply a model on the dataset with the configuration""")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-f", "--path_to_model", help="The absolute path to the model to predict or continue training.")
    parser.add_argument("-s", "--split_name", help="""The split name to perform the prediction or training on. This is required for predict. For example 'wl-c11'.""")
    parser.add_argument("-sf", "--split_files", help="A whitespace separated list of file names. For wildlife training defaults to [target_train, target_dev, target_test]")
    parser.add_argument("-m", "--do_multiclass", action="store_true", default=False)
    parser.add_argument("-l", "--inmemory", action="store_true", default=False, help="Whether to load all data into memory before operation.")
    parser.add_argument("-d", "--dryrun", action="store_true")
    
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
    
    model_type = config.getModelType()
    if not model_type:
        raise Exception("Please configure the model type and retry.")
    
    dataset_dir = config.getDatasetDirectoryPath()
    if run_opts.command == "training":
        if model_type == "baseline":
            start_training_baseline_from_config(config, dataset_dir, split_name, do_multiclass=run_opts.do_multiclass)
    
    if run_opts.command == "evaluate":
        if not run_opts.path_to_model:
            raise Exception("Please provide the path to the model using the '-f' option and retry.")
        if model_type == "baseline":
            split_file_test = "target_test"
            if run_opts.split_files:
                split_file_test = run_opts.split_files.split(" ")[0]  # expect only a single file here for now
            if run_opts.inmemory:
                start_evaluate_baseline_in_memory(run_opts.path_to_model, dataset_dir, split_name, split_file_test, do_multiclass=run_opts.do_multiclass)
            else:
                start_evaluate_baseline(run_opts.path_to_model, dataset_dir, split_name, split_file_test, do_multiclass=run_opts.do_multiclass)

            
if __name__ == '__main__':
    main()
    
