'''

NOT IN USE FOR NOW

Created on 12.05.2019

@author: Philipp
'''

import tensorflow as  tf
from wildlife import get_dimensions, is_multiclass


def __keras_z_log(batch):
    log_batch = tf.keras.backend.log(batch)
    z_batch = (log_batch - tf.keras.backend.mean(log_batch)) / tf.keras.backend.std(log_batch)
    return z_batch


def create_adapter_model(save_path, layer_names, number_of_outputs, capacity=100, use_zlog=False, freeze_bottom=False):
    base_model = tf.keras.models.load_model(save_path)
    
    if freeze_bottom:
        for layer in base_model.layers:
            layer.trainable = False
    
    pred_model = base_model.layers[-1].output
    if use_zlog:
        print("Add z-log layer")
        pred_model = tf.keras.layers.Lambda(__keras_z_log, name="zlog")(pred_model)

    for layer_name in layer_names[:-1]:
        pred_model = tf.keras.layers.Dense(capacity, activation="elu", name=layer_name)(pred_model)
    pred_model = tf.keras.layers.Dense(number_of_outputs, activation="softmax", name=layer_names[-1])(pred_model)

    model = tf.keras.Model(base_model.input, pred_model) 
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model


def load_adapter_model(save_path, pred_model_path, layer_names, number_of_outputs, capacity=100, use_zlog=False):
    base_model = tf.keras.models.load_model(save_path)
    
    pred_model = base_model.layers[-1].output
    if use_zlog:
        print("Add z-log layer")
        pred_model = tf.keras.layers.Lambda(__keras_z_log)(pred_model)

    for layer_name in layer_names[:-1]:
        pred_model = tf.keras.layers.Dense(capacity, activation="elu", name=layer_name)(pred_model)
    pred_model = tf.keras.layers.Dense(number_of_outputs, activation="softmax", name=layer_names[-1])(pred_model)

    model = tf.keras.Model(base_model.input, pred_model) 
    model.load_weights(pred_model_path, by_name=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="categorical_crossentropy")
    
    return model


def create_prediction_model(x_train, y_train_cat):
    """
        Create a prediction model. This can be trained on outputs of the focus model.
    """
    number_of_inputs = get_dimensions(x_train)
    number_of_outputs = get_dimensions(y_train_cat)

    capacity = 10
    learning_rate = 0.001
    if is_multiclass(y_train_cat):
        capacity = 100
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(capacity, activation="elu", name="adapter_1", input_dim=number_of_inputs))
    model.add(tf.keras.layers.Dense(capacity, activation="elu", name="adapter_2"))
    if is_multiclass(y_train_cat):
        model.add(tf.keras.layers.Dense(capacity, activation="elu", name="adapter_3"))
        model.add(tf.keras.layers.Dense(capacity, activation="elu", name="adapter_4"))
    model.add(tf.keras.layers.Dense(number_of_outputs, activation="softmax", name="adapter_out"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])
    return model
