'''
Created on 03.05.2019

@author: Philipp
'''
import numpy as np

import tensorflow as tf
from wildlife.dataset.wildlife.results import WildlifeResults


def start_prediction(config, path_to_model, source_split, target_split):
    results = WildlifeResults.create(config, source_split)
    with tf.Session():
        # The following are loaded as 'flat' files on the top directory
        prediction_sequence = None
        
        model = None #__get_model(config, path_to_model, 1)
        
        dryrun = config.is_dryrun()
        processed_count = 0
        expected_num_batches = len(prediction_sequence)
        try:
            for batch_inputs, batch_questions in prediction_sequence.one_shot_iterator():
                batch_predictions = model.predict_on_batch(batch_inputs)
                results.add_batch(batch_questions, batch_predictions)
                processed_count = processed_count + 1
                print(">> Processing batches {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
                if dryrun and processed_count > 10:
                    raise Exception("Dryrun finished")
        except:
            print("Processed all images: {}".format(processed_count))
        
    results.write_results_file(path_to_model, target_split)
    results.write_human_results_file(path_to_model)



def prediction_evaluate(model, x_test, y_true, title_mappings, verbose=0):
    """
        Run the model against a test dataset to create prediction results. 
        Evaluate the prediction results. Returns the prediction results.
    """
    y_hat = apply_pred_model(model, x_test, y_true, verbose)
    print_metrics(y_hat, y_true, title_mappings) 
    return y_hat


def predict_on(model, x_dataset, return_transpose=True, verbose_level=0):
    predictions = model.predict(x_dataset, 100, verbose=verbose_level)
    predictions = np.squeeze(predictions)
    if return_transpose:
        predictions = np.transpose(predictions)
    return predictions


def z_log(predictions, verbose=False):
    log_predictions = np.log(predictions)
    z_predictions = (log_predictions - np.mean(log_predictions)) / np.std(log_predictions)
    if verbose:
        print("Predictions: '[%s]'" % ', '.join(map(str, predictions[:5])))
        print("log-predictions: '[%s]'" % ', '.join(map(str, log_predictions[:5])))
        print("z-log-predictions: '[%s]'" % ', '.join(map(str, z_predictions[:5])))
    return z_predictions


def apply_pred_model(pred_model, x_dataset, y_dataset, verbose=0):
    predictions = predict_on(pred_model, x_dataset, return_transpose=False, verbose_level=verbose)
    return analyse_results(predictions, y_dataset)
    
def analyse_results(predictions, y_dataset):
    prediction_classes = np.round(predictions).astype(np.int32)
    prediction_classes = np.squeeze(prediction_classes)
    
    prediction_classes = __shrink(prediction_classes)
    y_dataset = __shrink(y_dataset)

    total_images = len(predictions)
    matches = np.equal(prediction_classes, y_dataset)
    matches = matches.astype(np.int32)
    total_matches = np.sum(matches)
    print("Matches: {:.2} {}/{}".format(total_matches / total_images, total_matches, total_images))

    idx_mismatches = np.argwhere(matches == 0)
    idx_mismatches = np.squeeze(idx_mismatches)
    len("Mismatches: {}".format(idx_mismatches))

    total_mismatches = len(idx_mismatches)
    y_mismatches = y_dataset[idx_mismatches]
    prediction_mismachtes = np.squeeze(prediction_classes[idx_mismatches])

    false_positives = np.sum(np.squeeze(np.greater(prediction_mismachtes, y_mismatches).astype(np.int32)))
    false_negatives = np.sum(np.squeeze(np.greater(y_mismatches, prediction_mismachtes).astype(np.int32)))
    print("False positives: {:.2} {} / {}".format(false_positives / total_mismatches, false_positives, total_mismatches))
    print("False negatives: {:.2} {} / {}".format(false_negatives / total_mismatches, false_negatives, total_mismatches))

    print("False positives (total): {:.2} {} / {}".format(false_positives / total_images, false_positives, total_images))
    print("False negatives (total): {:.2} {} / {}".format(false_negatives / total_images, false_negatives, total_images))
    print()
    return prediction_classes


def apply_pred_model_argmax(pred_model, x_dataset):
    predictions_sparse = predict_on(pred_model, x_dataset, return_transpose=False)
    predictions_sparse = np.argmax(predictions_sparse, axis=1).astype(np.int32)
    predictions_sparse = np.squeeze(predictions_sparse)
    return predictions_sparse


def print_metrics(predictions, y_dataset, ids_to_labels):
    predictions = __shrink(predictions)
    y_dataset = __shrink(y_dataset)
    
    print("Overall predicted classes: {}".format([(idx, ids_to_labels[idx]) for idx in np.unique(predictions)]))
    print("Overall  existing classes: {}".format([(idx, ids_to_labels[idx]) for idx in np.unique(y_dataset)]))
    
    __print_accuracy(predictions, y_dataset, ids_to_labels)
    __print_confusion_matrix(predictions, y_dataset, ids_to_labels)
    __print_report(predictions, y_dataset, ids_to_labels)


from sklearn.metrics import accuracy_score


def __print_accuracy(predictions, y_dataset, title_mappings):
    """
        Calculate and print precision, recall, f1-score. 
        
        predictions : predictions e.g. [4, 1]
        y_dataset : true labels e.g. [4, 1]
        title_mappings: dict of ids_to_labels e.g. {1:"a", 4:"b"}
    """
    score = accuracy_score(y_dataset, predictions)
    print("Accuracy: {:.2}".format(score))
    print()

    
from sklearn.metrics import confusion_matrix


def __print_confusion_matrix(predictions, y_dataset, title_mappings):
    """
        Calculate and print confusion matrix. 
        
        predictions : predictions e.g. [4, 1]
        y_dataset : true labels e.g. [4, 1]
        title_mappings: dict of ids_to_labels e.g. {1:"a", 4:"b"}
    """
    confusion = confusion_matrix(predictions, y_dataset, labels=[k for k in title_mappings])

    print("{:15}".format("pred v / true>"), end="")
    for label in title_mappings.values():
        print("{}".format(label), end="  ")
    print()
    for row_cls, pred_counts in enumerate(confusion):
        row_header = title_mappings[row_cls]
        print("{:15}".format(row_header), end="")
        for col_header_idx, pred_count in enumerate(pred_counts):
            col_header = title_mappings[col_header_idx]
            print("{count:>{pad}}".format(pad=len(col_header), count=pred_count), end="  ")
        print()
    print()

        
from sklearn.metrics import classification_report


def __print_report(predictions, y_dataset, title_mappings):
    """
        Calculate and print precision, recall, f1-score. 
        
        predictions : predictions e.g. [4, 1]
        y_dataset : true labels e.g. [4, 1]
        title_mappings: dict of ids_to_labels e.g. {1:"a", 4:"b"}
    """
    r = classification_report(y_dataset, predictions,
                          labels=[k for k in title_mappings],
                          target_names=[v for v in title_mappings.values()])
    print(r)
    print()

    
def __shrink(y_dataset):
    # shrink categorical if necessary
    y_datast_shape = np.shape(y_dataset)
    if len(y_datast_shape) > 1:
        y_dataset = np.argmax(y_dataset, axis=1)
        # print(y_datast_shape, " shrinks to ", np.shape(y_dataset))
    return y_dataset   
