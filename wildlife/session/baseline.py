'''
Created on 06.06.2019

@author: Philipp
'''
import numpy as np
import tensorflow as tf
from wildlife import to_split_dir
from wildlife.dataset.images.tfrecords import load_tfrecord_in_memory, \
    create_dataset_sample_op
from wildlife.model import focus
from wildlife.session import to_categorical
from wildlife.session.callbacks import create_tensorboard_from_dataset, \
    create_checkpointer
from wildlife.session.weights import calculate_class_weights
from wildlife.session.prediction import prediction_evaluate, analyse_results, \
    print_metrics


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

    dataset_split_dir = to_split_dir(dataset_dir, split_name)
    
    dataset_string = "Loading {} data into memory from {}".format(split_file_train, dataset_split_dir)
    print("\n{:-^80}".format(dataset_string))
    x_train, y_train, _ = load_tfrecord_in_memory(dataset_split_dir, split_file_train)
    
    dataset_string = "Loading {} data into memory from {}".format(split_file_validate, dataset_split_dir)
    print("\n{:-^80}".format(dataset_string))
    x_validate, y_validate, _ = load_tfrecord_in_memory(dataset_split_dir, split_file_validate)
    
    x_train = x_train / 255
    x_validate = x_validate / 255
    
    print("\n{:-^80}".format("Preparing training labels for {} classification problem".format("multi" if do_multiclass else "binary")))
    
    print("Labels ({}): {}".format(len(np.unique(y_train)), np.unique(y_train)))
    if do_multiclass:
        title_mappings, label_to_id = as_multiclass_problem()
    else:
        title_mappings, label_to_id = as_binary_problem()
        
    y_train_ids, y_train_cat = to_categorical(y_train, label_to_id)
    _, y_validate_cat = to_categorical(y_validate, label_to_id)
    
    class_weights = calculate_class_weights(y_train_ids, title_mappings)
    
    print("\n{:-^80}".format("Preparing model for training"))
    model_classifier = config.getModelClassifier()
    use_bn = config.getUseBatchNormalization()
    model = focus.create_model(model_classifier, y_train_cat, use_batch_norm=use_bn)
    
    print("\n{:-^80}".format("Preparing callbacks for training"))
    logdir = config.getTensorboardLoggingDirectory()
    log_path, tensorboard = create_tensorboard_from_dataset(logdir, model_classifier, do_multiclass, split_name)
    model_name = "wildlife-baseline-{}-{}".format(model_classifier, "bn" if use_bn else "no-bn")
    checkpointer = create_checkpointer(log_path, model_name)
        
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


def start_evaluate_baseline_in_memory(path_to_model,
                           dataset_dir, split_name,
                           split_file_test="target_test",
                           do_multiclass=True):

    dataset_split_dir = to_split_dir(dataset_dir, split_name)
    
    dataset_string = "Loading {} data into memory from {}".format(split_file_test, dataset_split_dir)
    print("\n{:-^80}".format(dataset_string))
    x_test, y_test, _ = load_tfrecord_in_memory(dataset_split_dir, split_file_test)
    x_test = x_test / 255
    
    print("\n{:-^80}".format("Preparing training labels for {} classification problem".format("multi" if do_multiclass else "binary")))
    
    print("Labels ({}): {}".format(len(np.unique(y_test)), np.unique(y_test)))
    if do_multiclass:
        title_mappings, label_to_id = as_multiclass_problem()
    else:
        title_mappings, label_to_id = as_binary_problem()
        
    _, y_test_cat = to_categorical(y_test, label_to_id)
    
    print("\n{:-^80}".format("Loading model from " + str(path_to_model)))
    model = tf.keras.models.load_model(path_to_model)
    
    print("\n{:-^80}".format("Start prediction"))
    prediction_evaluate(model, np.squeeze(x_test), np.squeeze(y_test_cat), title_mappings, verbose=1) 


def start_evaluate_baseline(path_to_model,
                           dataset_dir, split_name,
                           split_file_test="target_test",
                           do_multiclass=True):

    dataset_split_dir = to_split_dir(dataset_dir, split_name)
    
    dataset_string = "Loading {} data as iterator {}".format(split_file_test, dataset_split_dir)
    print("\n{:-^80}".format(dataset_string))
    sample_op = create_dataset_sample_op(dataset_split_dir, split_file_test, batch_size=100)
    
    print("\n{:-^80}".format("Preparing training labels for {} classification problem".format("multi" if do_multiclass else "binary")))
    
    if do_multiclass:
        title_mappings, label_to_id = as_multiclass_problem()
    else:
        title_mappings, label_to_id = as_binary_problem()
    
    print("\n{:-^80}".format("Start prediction"))
    results = []
    ground_truth_categorical = []
    with tf.Session() as sess:
        print("\n{:-^80}".format("Loading model from " + str(path_to_model)))
        model = tf.keras.models.load_model(path_to_model)
        processed_count = 0
        try:
            while True:
                processed_count = processed_count + 1
                print(">> Apply model on images {:d}".format(processed_count * 100), end="\r")
                images, labels, _ = sess.run(sample_op)
                images = images / 255
                _, labels_categorical = to_categorical(labels, label_to_id)
                ground_truth_categorical.extend(labels_categorical)
                result = model.predict_on_batch(images)
                results.extend(result)
        except Exception as e:
            print()
            print("Applied model on all images: {}".format(len(results)))
    
    results = np.array(results)
    results = np.squeeze(results)
    
    ground_truth = np.array(ground_truth_categorical)
    ground_truth = np.squeeze(ground_truth)
    
    prediction_classes = analyse_results(results, ground_truth)
    print_metrics(prediction_classes, ground_truth, title_mappings)
