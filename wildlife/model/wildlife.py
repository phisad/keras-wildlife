'''

NOT IN USE FOR NOW

Created on 12.05.2019

A model learning on the wildlife dataset. This is also often referenced
as a target model, because the wildlife dataset is the target dataset to
be classified.

A wildlife model can be initialized with an imagenet/focus model pre-trained
on a subset of imagenet categories. The wildlife model could also be trained
from scratch directly on the wildlife dataset.

The wildlife model is based on the VGG16 model likewise the imagenet model
with a classification network on top. The intermediate classification layer 
is put directly on the base model with a softmax classifier. The expert
groups are not supported here (because they havnt shown improvements, but 
come with more complexity).

After the possibly pre-trained base model with intermediate classifier an
adapter network is attached. The adapter network is supposed to perform 
the main transformation from the possibly pre-trained imagenet source domain
to the wildlife target domain. 

The adapter network on top is preprended with an optional batch normalization 
layer, which is recommended as experiments have shown the helpfulness for the
domain translation here. Then the adapter network has more capacity for the
multi classification task than for the binary task. The ultimate output is a
softmax classifier for the wildlife domain.  

@author: Philipp
'''
import tensorflow as  tf
from wildlife import get_dimensions, is_multiclass


def create_model(y_train_cat, focus_model_path=None, use_batch_norm=True):
    """
        The model is compiled before return.
        
        @param y_train_cat: array
            The categorical training label ids to automatically determine the number of outputs.
        @param focus_model_path: str
            The path to a model already pre-trained on the imagenet dataset.
        @param use_batch_norm: boolean 
            If a batch normalization layer should be attached before the adapter network.
    """
    print("Create wildlife/target model")
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

    number_of_outputs = get_dimensions(y_train_cat)
    capacity = 10
    learning_rate = 0.00001
    
    if is_multiclass(y_train_cat):
        print("Increasing capacity, because multi-classification task is detected")
        capacity = 100

    if use_batch_norm:
        print("Adding batch normalization layer before adapter network")
        model = tf.keras.layers.BatchNormalization(name="adapter_bn")(model)    

    model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_1")(model)    
    model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_2")(model)

    if is_multiclass(y_train_cat):
        model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_3")(model)
        model = tf.keras.layers.Dense(capacity, activation="elu", name="adapter_4")(model)

    model = tf.keras.layers.Dense(number_of_outputs, activation="softmax", name="adapter_out")(model)
    model = tf.keras.Model(base_model.input, model)
    
    if focus_model_path:
        print("Try to load model from path: " + focus_model_path)
        model.load_weights(focus_model_path, by_name=True, compile=False)
    
    print("Compile model with {} optimizer, categorical loss and metrics".format("Adam"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])
    return model
