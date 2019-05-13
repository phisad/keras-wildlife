'''
Created on 13.05.2019

@author: Philipp
'''
from io import BytesIO
import sys
from wildlife.dataset.images import read_single_rgb
from PIL import Image

from multiprocessing import Pool
import tqdm


def load_and_preprocess_data_into_parallel(dicts, number_of_processes):
    results = []
    with Pool(processes=number_of_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(__load_and_preprocess_single_defended, dicts), total=len(dicts)):
            results.append(result)
    return results

        
def __load_and_preprocess_single_defended(imaged):
    try:
        __load_and_preprocess_single(imaged)
        return ("Success", imaged)
    except:
        err_msg = sys.exc_info()[0]
        err = sys.exc_info()[1]
        error = (imaged["path"], err_msg, err)
        return ("Failure", error)


def load_and_preprocess_data_into(dicts):
    total = len(dicts)
    counter = 0
    errors = []
    for d in dicts:
        counter += 1
        print('>> Load and preprocess image %d/%d' % (counter, total), end="\r")
        try:
            __load_and_preprocess_single(d)
        except:
            err_msg = sys.exc_info()[0]
            err = sys.exc_info()[1]
            errors.append((d["path"], err_msg, err))
    print()
    for error in errors:
        print(error)

        
def __load_and_preprocess_single(imaged):
    """
        Read RGB, resize smallest, crop largest
    """
    with read_single_rgb(imaged["path"]) as image:
        if not imaged["site"] == "imagenet":  # wildlife image
            image = __remove_status_bar_single(image)
        image = __resize_single_smallest_with_ratio(image, imaged["width"])
        image = __crop_single_largest(image)
        with BytesIO() as raw:
            image.save(raw, "JPEG")
            imaged["data"] = raw.getvalue()
        imaged["path"] = imaged["path"].encode()
        imaged["site"] = imaged["site"].encode()
        imaged["label"] = imaged["label"].encode()


def __remove_status_bar_single(image):
    width, height = image.size
    height = height - 64
    to_crop_box = (0, 0, width, height)
    image = image.crop(box=to_crop_box)
    return image


def __resize_smallest_dim_with_ratio(width, height, target):
    """
        Resize the smallest dimension while keeping the aspect ratio.
    """
    if height == width:
        return target, target
    if height < width:
        rwidth = (width / height) * target
        return int(rwidth), target 
    if width < height:
        rheight = (height / width) * target
        return target, int(rheight)
    return width, height


def __resize_single_smallest_with_ratio(image, target):    
    width, height = __resize_smallest_dim_with_ratio(*image.size, target)
    image = image.resize((width, height), Image.BILINEAR)
    return image


def __crop_largest_rectangle(width, height, verbose=False):
    if width == height:
        return 0, 0, width, height
    largest = width
    smallest = height
    if height > width:
        largest = height
        smallest = width
    to_crop = largest - smallest
    if verbose:
        print(to_crop)
    offset_largest_start = to_crop // 2
    offset_largest_end = largest - offset_largest_start
    if to_crop % 2 != 0:
        offset_largest_end = offset_largest_end - 1
    if width > height:
        return offset_largest_start, 0, offset_largest_end, height
    if height > width:
        return 0, offset_largest_start, width, offset_largest_end  
    return 0, 0, width, height    


def __crop_single_largest(image, verbose=False):
    """
        Returns a copy of the image that has the largest dimension dropped
    """
    width, height = image.size
    image = image.copy()
    if verbose:
        print(image.size)   
    to_crop_box = __crop_largest_rectangle(width, height)
    image = image.crop(box=to_crop_box)
    if verbose:
        print(image.size)  
    return image
