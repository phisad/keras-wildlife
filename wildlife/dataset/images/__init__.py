import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array


def _get_image(image_file_path, target_shape):
    with load_img(image_file_path, target_size=target_shape) as image:
        imagearr = img_to_array(image)
    return imagearr


def _get_image_paths(directory_path):
    return ["/".join([directory_path, file]) for file in os.listdir(directory_path) if file.endswith('.jpg')]
