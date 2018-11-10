# Analysis of the inc_angle parameter
# Visualization of dependence on the inc_angle, calculation of normalization parameters
import matplotlib.pyplot as plt
import json
import numpy
import random
import plotly.plotly as py 
import numpy as np


def get_class_angles(datafile, train = False):
    with open(datafile) as json_data:
        data = json.load(json_data)
    icebergs = []
    ships = []
    for (i, sample) in enumerate(data):
        if (not isinstance(sample["inc_angle"], float)):
            continue
    
        # do plotting on the training set
        if (train and sample["is_iceberg"]):
            plt.plot(i, sample["inc_angle"], 'ro', label = 'iceberg')
            icebergs.append(sample["inc_angle"])
        elif (train):
            plt.plot(i, sample["inc_angle"], 'bo', label = 'ship')
            ships.append(sample["inc_angle"])
        # for the test set just append the angles, no plots here
        else:
            icebergs.append(sample["inc_angle"])
            
    return icebergs, ships

# get angles for both classes and perform visualization
icebergs, ships = get_class_angles("../datasets/processed/train.json", True)

axes = plt.gca()
axes.set_ylim([36.05, 37.35])
plt.xlabel("Sample number")
plt.ylabel("Incidence angle")

histogram=plt.figure(2)

bins = numpy.linspace(30, 46, 321)
plt.hist(icebergs, bins, alpha=0.7, color='r', label = 'icebergs')
plt.hist(ships, bins, alpha=0.7, color='b', label = 'ships')
plt.legend(loc='upper right')
plt.xlabel("Incidence angle")
plt.ylabel("Number of samples")

plt.show()

# now do some analysis - compute mean and variance of all angles
angles = icebergs + ships

angles_test, _ = get_class_angles("../datasets/processed/test.json", False)
angles = angles + angles_test

angles = np.asarray(angles)
print("Mean angle:", np.mean(angles))
print("Standard deviation:", np.std(angles))

