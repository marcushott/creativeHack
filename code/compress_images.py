import PIL
from PIL import Image
import os

directory = '../datasets/img_pos/'
for imname in os.listdir(directory):
    print(imname[-3:])
    if (imname[-3:] == "jpg" or imname[-3:] == "png"):
        img = Image.open(directory + imname) # image extension *.png,*.jpg
        img = img.resize((112,112), PIL.Image.BILINEAR) 
        img.save('../datasets/img_raw/'+imname) 
