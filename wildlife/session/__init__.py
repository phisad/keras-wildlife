'''
Created on 01.03.2019

@author: Philipp
'''
from wildlife.dataset.wildlife.labels import convert_label_to_ids
import tensorflow as tf


def to_categorical(y_labels, label_to_id):
    #print("to_categorical", y_labels[:5])
    y_ids = convert_label_to_ids(y_labels, label_to_id)
    #print("to_categorical", y_ids[:5])
    y_categorical = tf.keras.utils.to_categorical(y_ids)
    #print("to_categorical", y_categorical[:5])
    return y_ids, y_categorical

