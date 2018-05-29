#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:59:23 2018

@author: kaku
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2, os, datetime
import keras
import keras.backend as K
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model, model_from_json

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
def save_model(model, model_path, file_time_global):
    """
    Save model into model_path
    """
    json_string=model.to_json()
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    modle_path=os.path.join(model_path,'architecture'+'_'+file_time_global+'.json')
    open(modle_path,'w').write(json_string)
    model.save_weights(os.path.join(model_path,'model_weights'+'_'+ file_time_global+'.h5'),overwrite= True )
    
def read_model(model_path, file_time_global):
    """
    Read model from model_path
    """
    model=model_from_json(open(os.path.join(model_path,'architecture'+'_'+ file_time_global+'.json')).read())
    model.load_weights(os.path.join(model_path,'model_weights'+'_'+file_time_global+'.h5'))
    return model

def slide_window(img, win ,stride):
    standrad = 28, 28
    img_h, img_w = img.shape
    win_h, win_w = win
    stride_h, stride_w = stride
    patchs = []
    idxs = []
    for i in range(int(np.floor((img_h-win_h)/stride_h))):
        for j in range(int(np.floor((img_w-win_w)/stride_w))):
            patch = img[i*stride_h:(i*stride_h+win_h),j*stride_w:(j*stride_w+win_w)]
            patch = cv2.resize(patch, standrad)
            patchs.append(patch.reshape(1, standrad[0], standrad[0], 1))
            idxs.append(i*np.floor((img_h-win_h)/stride_h))
    size_new = int(np.floor((img_h-win_h)/stride_h)), int(np.floor((img_w-win_w)/stride_w))
    return patchs, idxs, size_new

K.set_image_data_format('channels_last')

time_global = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
script_name = os.path.basename(__file__)

model_path = '../model/'

img_rows, img_cols = 28, 28
win = 40, 20
stride = 10, 10
num_class = 10
epochs = 100
batch_size = 128
data_aug = True
model_train = False
model_name = '2018-05-29-15-58'

if model_train:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32).reshape(-1,img_rows,img_cols,1)/255
    y_train = to_categorical(y_train, 10)
    x_test = x_test.astype(np.float32).reshape(-1,img_rows,img_cols,1)/255
    y_test = to_categorical(y_test, 10)
    
    input_shape = x_train.shape[1:]
    model = mnist_model(input_shape)
    model_name = time_global
    if data_aug:
        datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)        
        datagen.fit(x_train,augment=True)  
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
    save_model(model, model_path, model_name)
    
else:
    model = read_model(model_path, model_name)
    
#x_test = cv2.imread('./sample.png',0).astype(np.float32)/255
#x_test = cv2.resize(x_test, (28,28)).reshape(1,28,28,1)
#x_test = - (x_test - 1)
#
#rotate_x_test = np.transpose(x_test)


img = cv2.imread('../imainaka.png',0).astype(np.float32)/255
#x_test = cv2.resize(x_test, (28,28)).reshape(1,28,28,1)
img = - (img - 1)

#rotate_x_test = np.transpose(x_test)

a, b, size_new = slide_window(img, win ,stride)
a_imgs = np.concatenate(a)

r = model.predict(a_imgs)

r_max = np.argmax(r,axis = 1)

result = np.reshape(r_max,size_new)

plt.imshow(result)
plt.imsave('../result/result.png', result)



    


