'''
Created on 03.05.2019

@author: Philipp
'''


def tuples_to_dicts_from_config(config, tuples):
    target_shape = config.getImageInputShape()
    wildlife_path = config.getWildlifeDatasetDirectoryPath()
    return tuples_to_dicts(tuples, wildlife_path, target_shape)
    
def tuples_to_dicts(tuples, wildlife_path="/data/wildlife/", target_shape=(224, 224)):
    # data, path, site, height, width, label
    dicts = []
    for tup in tuples:
        d = {}
        path = tup[0]
        label = tup[1][0]
        d["path"] = path
        d["label"] = label
        d["site"] = "imagenet"
        if path.startswith(wildlife_path):
            d["site"] = __get_site(path[len(wildlife_path):])
        d["width"] = target_shape[0]
        d["height"] = target_shape[1]
        dicts.append(d)
    return dicts


def __get_site(filepath):
    filepath = filepath.replace("\\", "/")
    filepath = filepath.split("/")[:-1]
    return "/".join(filepath)

