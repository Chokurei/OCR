#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:53:24 2018

@author: kaku
"""

from keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image


img_generator = ImageDataGenerator(
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.3
    )

img = []
band = np.arange(900).reshape(30,30,1)
for i in range(3):
    img.append(band)
img = np.concatenate(img,-1)

x = image.img_to_array(img)
x = np.expand_dims(img, 0)
gen = img_generator.flow(x, batch_size=1)

plt.figure()
for i in range(3):
    for j in range(3):
        x_batch = next(gen)
        idx = (3*i) + j
        plt.subplot(3,3,idx+1)
        plt.imshow(x_batch[0]/256)
        
x_batch.shape
