'''
Created on 12.05.2019

A model possibly trained on an imagenet subset before, then trained on the wildlife dataset.
For the baseline, this model is also trained directly on the wildlife dataset.

This is also referenced as a focus model, because in this way the model pre-trained 
on a subset of the imagenet categories that are related to the wildlife categories.

The base model is VGG16 on which a classification layer is put. The classification
layer are either groups of experts or a softmax classifier.

A group of experts on top is dedicated to a single category. Therefore, with experts
the model has as many outputs as expert groups. These are optimized using binary cross
entropy, meaning that each group is independently optimized to detect their category.

A softmax layer on top is applied for all categories at once. Therefore, the model
only has a single n-dimensional output, one dimension for each category.

@author: Philipp
'''
import tensorflow as tf
from wildlife import get_dimensions


def create_model(model_type, y_train_cat, title_mappings=None, use_bn=True):
    """
        The model is compiled before return.
        
        @param model_type: str
            Either "experts" or "softmax" see description above.
        @param y_train_cat: array
            The categorical training label ids to automatically determine the number of outputs.
        @param title_mappings: dict (for experts only)
            The mapping from category id to label name to name the expert outputs.
        @param use_bn: boolean (for softmax only)
            If a batch normalization layer should be attached before the softmax classifier.
    """
    print("Create imagenet/focus model")

    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="avg",
        classes=None)
    
    number_of_outputs = get_dimensions(y_train_cat)
    print("Determined number of outputs: " + str(number_of_outputs))
    
    if model_type == "experts":
        experts_models = []
        for expert_group in range(number_of_outputs):
            expert_group = title_mappings[expert_group]
            experts_model = base_model.layers[-1].output
            experts_model = tf.keras.layers.Dense(8, activation="elu", name="{}_experts".format(expert_group))(experts_model)
            experts_model = tf.keras.layers.Dense(1, activation="sigmoid", name="{}".format(expert_group), use_bias=False)(experts_model)
            experts_models.append(experts_model)

        print("Compile EXPERTS model with {} optimizer, categorical loss and metrics".format("Adam"))
        model = tf.keras.Model(base_model.input, experts_models)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="binary_crossentropy", metrics=['accuracy'])

    if model_type == "softmax":
        top_model = base_model.layers[-1].output
        if use_bn:
            print("Adding batch normalization layer before softmax classifier")
            top_model = tf.keras.layers.BatchNormalization(name="output_bn")(top_model)    
        top_model = tf.keras.layers.Dense(number_of_outputs, activation="softmax", name="output")(top_model)
        
        print("Compile SOFTMAX model with {} optimizer, categorical loss and metrics".format("Adam"))
        model = tf.keras.Model(base_model.input, top_model)       
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="categorical_crossentropy", metrics=['accuracy'])
        
    return model
