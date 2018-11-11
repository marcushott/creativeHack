"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import cv2
from skimage import img_as_ubyte 
from skimage import img_as_float
import json
from pprint import pprint

import _pickle as pickle

def get_image_from_name(imname):
    #print(imname)
    return cv2.imread("../datasets/img_raw/"+imname).astype(dtype="float32") 

def compute_norm_params(data_file):
    with open(data_file) as json_data:
        data = json.load(json_data)

    # compute min and max over the training set for normalization
    mean, dev = np.zeros(3), np.zeros(3)
    ch = ([],[],[])
    
    for sample in data:
        img = get_image_from_name(sample["img"])
        #print(type(img[0,0,0]))
        mean += img.mean(axis=(0,1))
        for i in range(3):
            ch[i].append(img[:,:,i].flatten())
        
    #c = [np.concatenate(ch[i]) for i in range(3)]
    dev = [np.std(np.concatenate(ch[i])) for i in range(3)] 
    
    return np.asarray(mean, dtype="float")/len(data), np.asarray(dev, dtype="float")


class OutfitData(data.Dataset):
    def __init__(self, data_file, transform = None, norm_transform = None):
        self.root_dir_name = os.path.dirname(data_file)
        
        self.transform = transform
        self.norm_transform = norm_transform
        
        # self.data is a list of dictionaries
        # each element has the following fields (id, r, c, img, is_good)
        # id: image name, r: number of rows, c: number of coumns, img: array  
        # img: array r*c*3, is_good: {0:bad,1:good}
        
        with open(data_file) as json_data:
            self.data = json.load(json_data)
            
        json_data.close()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.data)

    def get_item_from_index(self, index):
        
        # rescale each band separately first in order to apply cv2 functionality
        #band1lp = np.float32(band1)
        
        #target_label = np.float32(self.data[index].get("label", -1))
        target_label = self.data[index].get("label", -1)
        img = get_image_from_name(self.data[index].get("img", -1))
        #if (target_label == -1):
        #    target_label = self.data[index]["id"]    
        #print(img.shape)
        
        img = np.transpose(img, (2, 0, 1))
        
        img = torch.from_numpy(img.copy())
        
        if self.transform:
            img = self.transform(img)
        
        # normalize if wanted
        if self.norm_transform:
            img = self.norm_transform(img)

        return img, target_label

    # rescale the image to [0, 1]
    def normalize_image(self, img):
        #band = np.array(band)
        cmin, cmax = np.min(img), np.max(img)
        img = (img - cmin)/(cmax-cmin)
        return img

def rel_error(x, y):
    """ Returns relative error """
    assert x.shape == y.shape, "tensors do not have the same shape. %s != %s" % (x.shape, y.shape)
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
