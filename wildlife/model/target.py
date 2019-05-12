'''
Created on 12.05.2019

@author: Philipp
'''
import tensorflow as  tf
from wildlife import __get_dimensions, __is_multiclass


def create_target_model(y_train_cat, focus_model_path=None, use_bn=False):
    """ model directly learning an the wildlife dataset """
    number_of_intermediate_outputs = 16
    
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="avg",
        classes=None)
    model = tf.keras.layers.Dense(number_of_intermediate_outputs,
                                        activation="softmax", name="output")(base_model.layers[-1].output)

    number_of_outputs = __get_dimensions(y_train_cat)
    capacity = 10
    learning_rate = 0.00001
    
    if __is_multiclass(y_train_cat):
        capacity = 100

    if use_bn:
        model = tf.keras.layers.BatchNormalization(name="adapter_bn")(model)    

    model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_1")(model)    
    model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_2")(model)

    if __is_multiclass(y_train_cat):
        model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_3")(model)
        model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_4")(model)

    model = tf.keras.layers.Dense(number_of_outputs, activation="softmax", name="adapter_out")(model)
    model = tf.keras.Model(base_model.input, model)
    
    if focus_model_path:
        model.load_weights(focus_model_path, by_name=True)
        
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])
    return model
