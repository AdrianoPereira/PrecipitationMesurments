# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:34:05 2020

@author: gustav

This script contains examples of how to prepare data and train QRNN models that implement MLP or CNN architecture.
"""
from data_loader_preprocess import preprocess_data
import extendedQRNN
import numpy as np
'''
Example 1:
    MLP model with the following parameters:
        layers = 8
        neurons in each layer = 256
        activation function = relu
'''

# load the data from wherever you have stored it
xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy') 

# preprocess and format the data for the MLP
model_name = 'MLP'
test_set_size = 0.5
train_set_size = 0.4
val_set_size = 0.1
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(xData, yData,times,distance, model_name, train_set_size, val_set_size)

model = extendedQRNN.QRNN((28*28*2+4,),quantiles, depth = 8,width = 256, activation = 'relu', model_name = 'MLP')

model.fit(x_train = x_train,
          y_train = y_train[:,3,3],
          x_val = x_val,
          y_val = y_val[:,3,3],
          batch_size = 512,
          maximum_epochs = 500)


model.save('MLP_model.h5')

'''
Example 2:
    CNN model. The layout can be found in the extndedQRNN file lines 480 to 527.
    If tou want to change the layers, neuron count etc. you have to manually
    do that in the extendecQRNN file starting at line 480.
'''

# load the data from wherever you have stored it
# xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
# yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
# times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
# distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')
#
# # preprocess and format the data for the MLP
# model_name = 'CNN'
# test_set_size = 0.5
# train_set_size = 0.4
# val_set_size = 0.1
# quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
#
# x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(xData, yData,times,distance, model_name, train_set_size, val_set_size)
#
# model = extendedQRNN.QRNN(input_dim = (28,28,2), quantiles = quantiles ,model_name ='CNN')
#
# model.fit(x_train = x_train,
#           y_train = y_train[:,3,3],
#           x_val = x_val,
#           y_val = y_val[:,3,3],
#           batch_size = 512,
#           maximum_epochs = 500)

