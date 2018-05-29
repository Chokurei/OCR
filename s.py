#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 20:38:44 2018

@author: kaku
"""
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras import backend as K
from keras.utils import to_categorical

batch_size = 128
epochs = 3
num_classes = 10

img_rows, img_cols =32, 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#if K.image_data_format == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
#    input_shape = (3, img_rows, img_cols)
#else:
#    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
#    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
#    input_shape = (img_rows, img_cols, 3)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape=x_train.shape[1:]

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), 
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))



