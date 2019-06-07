'''
Created on 12.05.2019

@author: Philipp
'''
import tensorflow as tf
import time


def create_tensorboard_from_dataset(logdir, model_type, is_multiclass, dataset_name):
    model_mode = "multiclass" if is_multiclass else "binary"
    model_log_path = "wildlife/{}/{}/{}".format(dataset_name, model_type, model_mode)
    return create_tensorboard(logdir, model_log_path)

        
def create_tensorboard(base_path, model_log_path):
    """
        @param model_type: softmax, experts, pretrained, scratch, simple
        @param model_mode: ml (merge learning), dl (direct learning), binary, multiclass
        @param dataset_name e.g. in-c14
    """
    time_tag = time.strftime("%H-%M-%S", time.gmtime())
    
    tagged_log_path = "{}/{}/{}".format(base_path, model_log_path, time_tag)
    print("- Tensorboard log: " + tagged_log_path)
    
    """ The order of lines in the metadata file is assumed to match the order of vectors in the embedding variable """
    tensorboard_logger = tf.keras.callbacks.TensorBoard(log_dir=tagged_log_path, write_graph=True)
    return tagged_log_path, tensorboard_logger


def create_checkpointer(tagged_log_path, model_name, store_per_epoch=False):
    """
        Create checkpoint callback based on a given log_path. 
    """
    if store_per_epoch:
        model_name = model_name + ".{epoch:03d}.h5"
    else:
        model_name = model_name + ".h5"
    model_path = "/".join([tagged_log_path, model_name])
    print("- Checkpoint monitor: max [val_acc] at " + model_path)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                      monitor="val_acc",
                                                      mode="max",
                                                      save_best_only=True,
                                                      save_weights_only=False,
                                                      verbose=1,
                                                      period=1)
    return checkpointer
