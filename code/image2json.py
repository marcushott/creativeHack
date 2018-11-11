#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import json
import glob
from random import shuffle

#load all images into the image_list
good_outfit_list = []
#load all images into the image_list
for images in glob.glob('../datasets/img_pos/*.jpg'): 
    name_image = images[20:]
    print(name_image)
    good_outfit_list.append(name_image)
    
bad_outfit_list = []
#load all images into the image_list
for images in glob.glob('../datasets/img_neg/*.jpg'): 
    name_image = images[20:]
    bad_outfit_list.append(name_image)
#concatenate outfit lists
outfits = good_outfit_list + bad_outfit_list

#generate labels
label_positive = [1]* len(good_outfit_list)
label_negative = [0]* len(bad_outfit_list)
labels = label_positive + label_negative

#zip labels and outfits before shuffeling
image_and_labels = list(zip(labels,outfits))
shuffle(image_and_labels)
labels, outfits = zip(*image_and_labels)
    

#assign train, validate and test set
number_images = len(outfits)
step1 = int(0.85*number_images)
step2 = number_images-1#int(0.95*number_images)

train_labels = labels[:step1]

validate_list = outfits[step1:step2]
validate_labels = labels[step1:step2]

test_list = outfits[step2:]
test_labels = labels[step2:]

outfits = outfits[:step1]


#add images to train, validate and test json object
train_data = [{"img": image,
            "label": y} for image, y in zip(outfits, train_labels)] 
validate_data = [{"img": image,
            "label": y} for image,y in zip(validate_list, validate_labels)]
test_data = [{"img": image,
            "label": y} for image,y in zip(test_list, test_labels)]

#print(validate_data)
#print(train_data)
#print(test_data)
#encode json objects and write to files
with open('../datasets/train.json', 'w') as outfile:  
    json.dump(train_data, outfile)
with open('../datasets/validate.json', 'w') as outfile:  
    json.dump(validate_data, outfile)
with open('../datasets/test.json', 'w') as outfile:  
    json.dump(test_data, outfile)
    
    