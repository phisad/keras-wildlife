'''
Created on 11.05.2019

@author: Philipp
'''
import collections
from collections import OrderedDict


def get_sites(images, top=50, verbose=False):
    sites = collections.defaultdict(list)
    for filepath in images:
        sites[get_site(filepath)].append(filepath)
    
    if not top:
        if verbose:
            list_site_count(ordered_sites(sites))
        return sites
    
    sites = ordered_sites(sites)
    top_keys = [key for idx, key in enumerate(sites.keys()) if idx < top]
    top_sites = {}
    for key in top_keys:
        top_sites[key] = sites[key]
    if verbose:
        list_site_count(ordered_sites(top_sites))
    return top_sites


def get_site(filepath):
    filepath = filepath.replace("\\", "/")
    filepath = filepath.split("/")[:-1]
    return "/".join(filepath)


def ordered_sites(sites):
    return OrderedDict(sorted(sites.items(), key=lambda t:-len(t[1])))


def list_site_count(sites):
    for idx, (site, images) in enumerate(sites.items()):
        print("{:2} {}: {}".format(idx + 1, site, len(images)))


def list_sites(sites):
    print("Sites: {}".format(len(sites)))

    for site, paths in OrderedDict(sorted(sites.items(), key=lambda t: len(t[1]))).items():
        print("{:10} : {:5}".format(site, len(paths)))
