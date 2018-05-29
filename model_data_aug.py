#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:59:23 2018

@author: kaku
"""

import numpy as np
import keras
import keras.backend as K
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

K.set_image_data_format('channels_last')

data_aug = True
img_rows, img_cols = 28, 28
num_class = 10
epochs = 10
batch_size = 128


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32).reshape(-1,img_rows,img_cols,1)/255
y_train = to_categorical(y_train, 10)
x_test = x_test.astype(np.float32).reshape(-1,img_rows,img_cols,1)/255
y_test = to_categorical(y_test, 10)

input_shape = x_train.shape[1:]

def mnist_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adadelta(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


model = mnist_model(input_shape)


if data_aug:
    datagen = ImageDataGenerator(
    #    featurewise_center=True,
    #    featurewise_std_normalization=True,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    
    datagen.fit(x_train,augment=True)
    
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    #                    steps_per_epoch=len(x_train)/batch_size, epochs=epochs)
    # x: tuple x[0][0]:[batch_size,img_cols,img_rows,bands]
    #           x[0][1]:[batch_size,num_class]
    #x = datagen.flow(x_train,y=y_train,batch_size=batch_size)
    
    gen = datagen.flow(x_train,y=y_train, batch_size = batch_size)
    x_augs = []
    y_augs = []
    for i in range(100):
        aug = next(gen)
        x_augs.append(aug[0])
        y_augs.append(aug[1])
    x_augs = np.concatenate(x_augs)
    y_augs = np.concatenate(y_augs)

    x_train = np.concatenate((x_train, x_augs))
    y_train = np.concatenate((y_train, y_augs))
    
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size,
          validation_data=(x_test, y_test))    

    
    
    





    


