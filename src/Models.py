# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:27:48 2020

@author: gustav
"""

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import Model,Sequential
from keras.layers import Conv2D, Input, MaxPooling2D, MaxPooling3D,UpSampling2D,Reshape ,Dropout,Conv2DTranspose,concatenate, Flatten, Dense,BatchNormalization,ZeroPadding2D
from keras.layers import ConvLSTM2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def down_block(inputs,kernels = 2,batch_norm = False):
    conv1 = Conv2D(kernels, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(kernels, 3, activation = 'relu', padding = 'same')(conv1)
    if batch_norm:
        bn = BatchNormalization()(conv1)
        return MaxPooling2D(pool_size=(2, 2))(bn), conv1
    
    return MaxPooling2D(pool_size=(2, 2))(conv1), conv1

def up_block(inputs,kernels, conv, batch_norm = False):
    #up6 = Conv2DTranspose(kernels, 2,strides=(2, 2), activation = 'relu', padding = 'same')(inputs)
    up6 = Conv2D(kernels, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(inputs))
   
    merge6 = concatenate([conv,up6])
    conv6 = Conv2D(kernels, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(kernels, 3, activation = 'relu', padding = 'same')(conv6)
    if batch_norm:
        bn = BatchNormalization()(conv6)
        return bn
    return conv6

def unet1(pretrained_weights = None,input_size = (256,256,1)):
   
    depth = 5
    inputs = Input(input_size)
    kernels = 32
    convs = []
    out = inputs
    for i in range(depth):
        print(kernels)
        out, conv = down_block(out,kernels,True)
        kernels = kernels*2
        convs.append(conv)
        
    conv5 = Conv2D(kernels, 3, activation = 'relu', padding = 'same')(out)
    out = Conv2D(kernels, 3, activation = 'relu', padding = 'same')(conv5)
    #out = Dropout(0.5)(out)
    #print(len(convs))
    for i in range(depth):
        kernels = kernels//2
        print(kernels)
        out= up_block(out,kernels, convs[-(i+1)], True)
        
    
    print(out.shape)
    #conv9 = Conv2D(kernels, , activation = 'relu')(out)

    
    conv = Conv2D(32, 2, strides = 2, activation = 'relu')(out)
    conv = Conv2D(32, 5, strides = 1, activation = 'relu')(conv)
    conv = Conv2D(32, 7, strides = 1, activation = 'relu')(conv)
    conv = Conv2D(32, 7, strides = 1, activation = 'relu')(conv)

    conv10 = Conv2D(5, 1)(conv)

    
    
   
    print(conv10.shape)
    model = Model(inputs = inputs, outputs = conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()


    return model
def unet(pretrained_weights = None,input_size = (256,256,1)):
    
    inputs = Input(input_size)
    
    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(64, 2, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(128, 2, activation = 'relu', padding = 'same')(conv3)
    conv3 = ZeroPadding2D(padding = ((0, 1), (0, 1)))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv4)
    conv4 = ZeroPadding2D(padding = ((0, 1), (0, 1)))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
  
    
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv5)
    #drop5 = Dropout(0.5)(conv5)


    up7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5))    
    merge7 = concatenate([conv4,up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 2, activation = 'relu', padding = 'valid')(conv7)

    up8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))    
    merge8 = concatenate([conv3,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 2, activation = 'relu', padding = 'valid')(conv8)

    up9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv2,up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    
    up10 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv1,up10])
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge10)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv10)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv10)
    
   # conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv10)
    conv10 = Conv2D(5, 1)(conv10)

    model = Model(inputs = inputs, outputs = conv10)
    model.summary()
    return model

def convLSTM():
    
    model = Sequential()
    
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         input_shape=(None, 28, 28, 2),
                         return_sequences=True,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4,
                         recurrent_dropout=0.2
                         ))
    
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    
    model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3),
                         return_sequences=True,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
    
    model.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
                         return_sequences=False,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   data_format='channels_last')) 
    
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(5, activation = None))
    
    print(model.summary())

    return model