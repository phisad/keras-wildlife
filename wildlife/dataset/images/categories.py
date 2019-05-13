'''
Created on 11.05.2019

@author: Philipp
'''
import numpy as np


def split_categories(categories):
    """
        from datasetutils import split_categories
        categories = [
            (large_mappings["deer"],      1000, [800, 200]),
            (large_mappings["humankind"], 1000, [800, 200])
        ]
        splits = split_categories(categories)
    """
    splits = [[] for _ in categories[0][2]]
    for image_tuples, total_count, split_count_listing in categories:
        # select random indicies
        indicies = range(len(image_tuples))
        selection_idx = np.random.choice(indicies, size=total_count, replace=False)
        # split index selection
        selection_split_idx = __split_to(selection_idx, split_count_listing)
        __add_to(splits, image_tuples, selection_split_idx)  
    return splits


def __split_to(listing, splitting):
    splits = []
    offset = 0
    for s in splitting:
        split = listing[offset:offset + s]
        splits.append(split)
        offset = offset + s
    return splits


def __add_to(listings, select_from_listing, listing_of_indicies):
    for idx, listing in enumerate(listings):
        indicies = listing_of_indicies[idx]
        selection = np.take(select_from_listing, indicies, axis=0)
        selection = selection.tolist()
        listing.extend(selection)
        
