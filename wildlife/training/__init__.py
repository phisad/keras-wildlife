'''
Created on 01.03.2019

@author: Philipp
'''
from wildlife.dataset.wildlife.labels import convert_label_to_ids
import tensorflow as tf


def to_categorical(y_labels, label_to_id):
    y_ids = convert_label_to_ids(y_labels, label_to_id)
    y_categorical = tf.keras.utils.to_categorical(y_ids)
    return y_ids, y_categorical

