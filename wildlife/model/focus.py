'''
Created on 12.05.2019

@author: Philipp
'''
import tensorflow as tf
from wildlife import __get_dimensions


def create_focus_model(model_type, y_train_cat, title_mappings=None, use_bn=False):
    """ model trained on an imagenet subset before, then trained on the wildlife dataset """
    
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="avg",
        classes=None)
    
    number_of_outputs = __get_dimensions(y_train_cat)
    
    if model_type == "experts":
        experts_models = []
        for expert_group in range(number_of_outputs):
            expert_group = title_mappings[expert_group]
            experts_model = base_model.layers[-1].output
            experts_model = tf.keras.layers.Dense(8, activation="elu", name="{}_experts".format(expert_group))(experts_model)
            experts_model = tf.keras.layers.Dense(1, activation="sigmoid", name="{}".format(expert_group), use_bias=False)(experts_model)
            experts_models.append(experts_model)

        model = tf.keras.Model(base_model.input, experts_models)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="binary_crossentropy", metrics=['accuracy'])

    if model_type == "softmax":
        top_model = base_model.layers[-1].output
        if use_bn:
            top_model = tf.keras.layers.BatchNormalization(name="output_bn")(top_model)    
        top_model = tf.keras.layers.Dense(number_of_outputs, activation="softmax", name="output")(top_model)
        model = tf.keras.Model(base_model.input, top_model)       
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="categorical_crossentropy", metrics=['accuracy'])
        
    return model
