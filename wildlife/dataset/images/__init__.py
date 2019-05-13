import os

from PIL import Image


def read_single_rgb(image_path):
    """
        images = [read_single_rgb(image_file) for image_file in image_files]
    """
    image = Image.open(image_path)
    image = image.convert("RGB")
    return image
