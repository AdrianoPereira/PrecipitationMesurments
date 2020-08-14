# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 19:07:47 2020

@author: gustav
"""

import numpy as np
from data_loader import getTrainingData
  



# create training data
xData, yData , times, distance = getTrainingData(dataSize = 5000,nmb_GPM_pass = 200, GPM_resolution = 3)   



#%% code for training the QRNN
import extendedQRNN
tmpXData = np.zeros((len(xData),28,28,2))

for i in range(2):
    #mean1 = np.mean(xData[:,i,:,:])
    #std1 = np.std(xData[:,i,:,:])
    #xData[:,i,:,:] =  (xData[:,i,:,:]-mean1)/std1

    tmpXData[:,:,:,i] = (xData[:,i,:,:]-xData[:,i,:,:].min())/(xData[:,i,:,:].max()-xData[:,i,:,:].min())
# preprocess the data in what way you like and plsit into training, validation and test
xData = tmpXData
xTrain = xData[:int(len(xData)*0.4)]
yTrain = yData[:int(len(xData)*0.4)]

xVal = xData[int(len(xData)*0.4):int(len(xData)*0.5)]
yVal = yData[int(len(xData)*0.4):int(len(xData)*0.5)]

xTest = xData[int(len(xData)*0.5):]
yTest = yData[int(len(xData)*0.5):]

print(xTrain.shape)
print(xVal.shape)
print(xTest.shape)

quantiles = [0.1,0.3,0.5,0.7,0.9]

# note that when training the CNN, depth, width and activation is not relevant. See the extendedQRNN file
model = extendedQRNN.QRNN((28*28*2+4,),quantiles, depth = 8,width = 256, activation = 'relu', model_name ='CNN')

model.fit(x_train = xTrain,
          y_train = yTrain,
          x_val = xVal,
          y_val = yVal,
          batch_size = 256,
          maximum_epochs = 500)
