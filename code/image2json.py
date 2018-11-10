#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
from random import shuffle

#load all images into the image_list
image_list = []
#load all images into the image_list
for images in glob.glob('data/*.jpg'): 
    name_image = images[5:]
    image_list.append(name_image)

#randomly shuffle the list
shuffle(image_list)
#assign train, validate and test set
number_images = len(image_list)
step1 = int(0.8*number_images)
step2 = int(0.9*number_images)
test_list = image_list[step1:step2]
validate_list = image_list[step2:]
image_list = image_list[:step1]

#add images to train, validate and test json object
train_data = [{"img": image,
            "label": 1} for image in image_list] 
validate_data = [{"img": image,
            "label": 1} for image in validate_list]
test_data = [{"img": image,
            "label": 1} for image in test_list]

print(validate_data)

#encode json objects and write to files
with open('train.json', 'w') as outfile:  
    json.dump(data, outfile)
with open('validate.json', 'w') as outfile:  
    json.dump(data, outfile)
with open('test.json', 'w') as outfile:  
    json.dump(data, outfile)
    
    