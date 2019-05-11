import tensorflow as tf
import numpy as np
import sys
import os

def tfrecord_inputs(tfrecord_file, directory, batch_size=100):
    dataset = make_dataset("/".join([directory, tfrecord_file + ".tfrecord"]))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    sample_op = iterator.get_next()
    return sample_op

def load_tfrecord_in_memory(tfrecord_file, directory):
    tf.reset_default_graph()
    inputs = tfrecord_inputs(tfrecord_file, directory)
    images_all = []
    labels_all = []
    infos_all = []
    with tf.Session() as sess:
        try:
            while True:
                images, labels, infos = sess.run(inputs)
                images_all.extend(images)
                labels_all.extend(labels)
                infos_all.extend(infos)
        except:
            print("Loaded all inputs: {}".format(len(images_all)))
    return np.array(images_all), np.array(labels_all), np.array(infos_all)

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(data, path, site, height, width, label):
    """
        Creates a TFRecord example image entry.
    Args:
        image_data: The image as numpy array
        image_format: The image format type e.g. jpg
        height and width: The image sizes
        class_id: The image label
    """
    return tf.train.Example(features=tf.train.Features(feature={
      'image/data': bytes_feature(data),
      'image/path': bytes_feature(path),
      'image/site': bytes_feature(site),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/label': bytes_feature(label)
  }))

def write_tfrecord(image_tuples, target_directory, filename):
    """
        Creates a TFRecord file from a map of image files paths to class ids
    Args:
        image_tuples: List of image tuples (data, path, height, width, site, label)
        target_directory: where to put the file
        filename: how to name the file
    """
    tf.reset_default_graph()
    
    #image_path = tf.placeholder(dtype=tf.string)
    #image_raw = tf.read_file(image_path)
        
    errors = []
    num_images = len(image_tuples)
    with tf.Session():
        with tf.python_io.TFRecordWriter("/".join([target_directory, filename])) as tfrecord_writer:
            processed_count = 0
            for image_tuple in image_tuples:
                # Show progress
                processed_count += 1
                print('>> Converting image %d/%d' % (processed_count, num_images), end="\r")
                
                try:
                    example = image_to_tfexample(
                        image_tuple["data"],
                        image_tuple["path"],
                        image_tuple["site"],
                        image_tuple["height"],
                        image_tuple["width"],
                        image_tuple["label"]
                    )
                    tfrecord_writer.write(example.SerializeToString())
                except:
                    err_msg = sys.exc_info()[0]
                    err = sys.exc_info()[1]
                    errors.append((image_tuple["path"], err_msg, err))
                
    print()            
    print("Errors: {}".format(len(errors)))
    if len(errors) > 0:
        for (error_file, info, err) in errors:
            print("Error one file: {} because: {} / {}".format(error_file, info ,err))

def read_sample(example_raw):
    """
        Read a single TFRecord example and converting into an image
    Args:
        The TFRecord example that represents an image
    """
    example = tf.parse_single_example(
        example_raw,
        features={
            'image/data': tf.FixedLenFeature([], tf.string),
            'image/path': tf.FixedLenFeature([], tf.string),
            'image/site': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/label': tf.FixedLenFeature([], tf.string),
        })
    
    image_height = tf.cast(example['image/height'], tf.int32)
    image_width = tf.cast(example['image/width'], tf.int32)
    image = tf.image.decode_image(example['image/data'], channels=3)
    image = tf.reshape(image, (image_height, image_width, 3))
    
    info = []
    info.append(("path", example['image/path']))
    info.append(("site", example['image/site']))
    info = tf.convert_to_tensor(info)
    
    label = example["image/label"]
    return image, label, info

def make_dataset(tfrecord_filepath):
    """
        Returns a dataset ready to process a TFRecord file
    """
    dataset = tf.data.TFRecordDataset(tfrecord_filepath)
    dataset = dataset.map(read_sample)
    return dataset 

def write_label_file(labels_to_class_names, dataset_dir, filename="labels.txt"):
    """
        Writes a file with the list of class names row-wise like "1:class1".
    Args:
        labels_to_class_names: A map of (integer) labels to class names.
        dataset_dir: The directory in which the labels file should be written.
        filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))
            
def read_label_file(dataset_dir, filename="labels.txt"):
    """Reads the labels file and returns a mapping from ID to class name.
    Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.
    Returns:
        A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
        lines = lines.split('\n')
        lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names
