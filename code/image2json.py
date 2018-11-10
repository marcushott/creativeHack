#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 14:08:09 2018

@author: Marcus
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import glob

#load all images into the image_list
image_list = []
#load all images into the image_list
for images in glob.glob('../datasets/img_raw/*.jpg'): 
    #load image into pixel_x * pixel_y * 3 array
    current_im = plt.imread(images)
    print(current_im.shape)
    image_list.append(current_im.tolist())
    
data = [{"id": image_id,
            "img": image,
            "label": 1} for image_id,image in zip(range(len(image_list)),image_list)] 
    
with open('json2.json', 'w') as outfile:  
    json.dump(data, outfile)