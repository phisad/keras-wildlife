'''
Created on 03.05.2019

@author: Philipp
'''


def tuples_to_dicts(tuples):
    # data, path, site, height, width, label
    dicts = []
    for tup in tuples:
        d = {}
        path = tup[0]
        label = tup[1][0]
        d["path"] = path
        d["label"] = label
        d["site"] = "imagenet"
        if path.startswith("/data/wildlife/"):
            d["site"] = __get_site(path[len("/data/wildlife/"):])
        d["width"] = 224
        d["height"] = 224
        dicts.append(d)
    return dicts


def __get_site(filepath):
    filepath = filepath.replace("\\", "/")
    filepath = filepath.split("/")[:-1]
    return "/".join(filepath)

