'''
Created on 03.05.2019

@author: Philipp
'''
import numpy as np


def prediction_evaluate(model, x_test, y_true, title_mappings, verbose=0):
    """
        Run the model against a test dataset to create prediction results. 
        Evaluate the prediction results. Returns the prediction results.
    """
    y_hat = __apply_pred_model(model, x_test, y_true, verbose)
    print_metrics(y_hat, y_true, title_mappings) 
    return y_hat


def __predict_on(model, x_dataset, return_transpose=True, verbose_level=0):
    predictions = model.predict(x_dataset, 100, verbose=verbose_level)
    predictions = np.squeeze(predictions)
    if return_transpose:
        predictions = np.transpose(predictions)
    return predictions


def __apply_pred_model(pred_model, x_dataset, y_dataset, verbose=0):
    predictions = __predict_on(pred_model, x_dataset, return_transpose=False, verbose_level=verbose)
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
