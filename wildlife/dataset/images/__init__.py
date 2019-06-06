import os

from PIL import Image
from wildlife.configuration import determine_file_path


def get_tfrecord_filename(split_name):
    return "{}.tfrecord".format(split_name)


def get_preprocessing_tfrecord_file(target_directory_path_or_file, split_name):
    try:
        return determine_file_path(target_directory_path_or_file, get_tfrecord_filename(split_name), to_read=True)
    except Exception:
        return None

    
def read_single_rgb(image_path):
    """
        images = [read_single_rgb(image_file) for image_file in image_files]
    """
    image = Image.open(image_path)
    image = image.convert("RGB")
    return image
