'''
Created on 06.06.2019

@author: Philipp
'''
import numpy as np
from wildlife import to_split_dir
from wildlife.dataset.images.tfrecords import load_tfrecord_in_memory
from wildlife.training import to_categorical
from wildlife.model import focus
from wildlife.training.weights import calculate_class_weights
from wildlife.training.callbacks import create_tensorboard_from_dataset, \
    create_checkpointer


def as_binary_problem():
    # relinking adaption and power adaption
    label_to_id = {
        "background"  :  0,
        "horse"       :  1,  # domestic animal
        "cat"         :  1,  # domestic animal
        "dog"         :  1,  # domestic animal
        "deer"        :  1,
        "marten"      :  1,
        "hare"        :  1,
        "bird"        :  1,
        "wildboar"    :  1,
        "racoon"      :  1,
        "fox"         :  1
    }  # Total: 11
    title_mappings = {0: 'background', 1: 'animal'}
    return title_mappings, label_to_id


def as_multiclass_problem():
    label_to_id = {
        "background"  :  0,
        "horse"       :  1,  # domestic animal
        "cat"         :  2,  # domestic animal
        "dog"         :  3,  # domestic animal
        "deer"        :  4,
        "marten"      :  5,
        "hare"        :  6,
        "bird"        :  7,
        "wildboar"    :  8,
        "racoon"      :  9,
        "fox"         : 10
    }  # Total: 11

    def id_to_label(label_to_id):
        return dict([(cls, label) for (label, cls) in label_to_id.items()])

    title_mappings = id_to_label(label_to_id)
    return title_mappings, label_to_id


def start_training_baseline_from_config(config, dataset_dir, split_name,
                                        split_file_train="target_train",
                                        split_file_validate="target_dev",
                                        do_multiclass=True):

    print("\n{:-^80}".format("Preparing datasets for training"))
    dataset_split_dir = to_split_dir(dataset_dir, split_name)
    
    print("\n{} {} {}".format("Loading training data into memory from:", dataset_split_dir, split_file_train))
    x_train, y_train, _ = load_tfrecord_in_memory(split_file_train, dataset_split_dir)
    
    print("\n{} {} {}".format("Loading validation data into memory from:", dataset_split_dir, split_file_train))
    x_validate, y_validate, _ = load_tfrecord_in_memory(split_file_validate, dataset_split_dir)
    
    x_train = x_train / 255
    x_validate = x_validate / 255
    
    print("Labels ({}): {}".format(len(np.unique(y_train)), np.unique(y_train)))
    if do_multiclass:
        title_mappings, label_to_id = as_multiclass_problem()
    else:
        title_mappings, label_to_id = as_binary_problem()
        
    y_train_ids, y_train_cat = to_categorical(y_train, label_to_id)
    _, y_validate_cat = to_categorical(y_validate, label_to_id)
    
    class_weights = calculate_class_weights(y_train_ids, title_mappings)
    
    print("\n{:-^80}".format("Preparing model for training"))
    model = focus.create_model("softmax", y_train_cat, use_bn=True)
    
    print("\n{:-^80}".format("Preparing callbacks for training"))
    logdir = config.getTensorboardLoggingDirectory()
    tensorboard, log_path = create_tensorboard_from_dataset(logdir, "softmax", do_multiclass, split_name)
    checkpointer = create_checkpointer(log_path, "wildlife-baseline-softmax-bn.h5")
        
    print("\n{:-^80}".format("Start training"))
    number_of_epochs = config.getEpochs()
    if do_multiclass:
        number_of_epochs = number_of_epochs * 10
        print("Increasing epochs for multi classification problem to " + str(number_of_epochs))
    dryrun = config.is_dryrun()
    model.fit(x=x_train,
              y=y_train_cat,
              batch_size=config.getBatchSize(),
              class_weight=class_weights,
              validation_data=(x_validate, y_validate_cat),
              validation_steps=None if not dryrun else 10,
              epochs=number_of_epochs if not dryrun else 1,
              steps_per_epoch=None if not dryrun else 10,
              verbose=2 if not dryrun else 1,
              callbacks=[tensorboard, checkpointer],
              use_multiprocessing=config.getUseMultiProcessing(),
              workers=config.getWorkers(),
              max_queue_size=config.getMaxQueueSize()
            )
            # initial_epoch=0 if not initial_epoch else initial_epoch)
