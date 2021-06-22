# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:37:30 2020

@author: gustav

script containing functions for preparing the data for the QRNN models
"""
import numpy as np

def preprocess_data(xData,yData,times,distance, model_name, train_set_size, val_set_size):
    
    # remove nan values
    
    nanValues =np.argwhere(np.isnan(xData)) 
    xData = np.delete(xData,np.unique(nanValues[:,0]),0)
    yData = np.delete(yData,np.unique(nanValues[:,0]),0)
    times = np.delete(times,np.unique(nanValues[:,0]),0)
    distance = np.delete(distance,np.unique(nanValues[:,0]),0)
    
    
    preprocessedXData = np.zeros((xData.shape[0],xData.shape[2],xData.shape[3],xData.shape[1]))
   
    # this section standardizes the inputs for the models. The MLP uses mean 0
    # variance 1 and the CNN is between 0 and 1. This can easily be adapted for
    # whatever preference you like.
    train_set_index = int(train_set_size*len(xData))
    val_set_index = int(train_set_size*len(xData)+val_set_size*len(xData))
    
    for i in range(2):
        
        if model_name == 'MLP':
            mean1 = np.mean(xData[:,i,:,:], axis = 0,keepdims = True)
            std1 = np.std(xData[:,i,:,:], axis = 0,keepdims = True)
            xData[:,i,:,:]  =  (xData[:,i,:,:]-mean1)/std1
        else:
            #preprocessedXData[:,:,:,i] = (xData[:,i,:,:]-xData[:train_set_index,i,:,:].min())/(xData[:train_set_index,i,:,:].max()-xData[:train_set_index,i,:,:].min())
            preprocessedXData[:,:,:,i] = (xData[:,i,:,:]-xData[:,i,:,:].min())/(xData[:,i,:,:].max()-xData[:,i,:,:].min())
    
    if model_name == 'MLP':
        newXData = np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]))
        tmp = np.zeros((xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]+4))
        tmp[:,:xData.shape[1]*xData.shape[2]*xData.shape[3]] = newXData
        
        # here is an example of how you can add time difference and distance 
        # differences to the MLP model as well
        tmp[:,-1] = (times[:,0]-times[:,1])/1000
        tmp[:,-2] = distance[:,0]
        tmp[:,-3] = distance[:,1]
        tmp[:,-4] = (times[:,0]-times[:,2])/1000
        preprocessedXData = tmp
        
    x_train = preprocessedXData[:train_set_index]
    y_train = yData[:train_set_index]
    
    x_val = preprocessedXData[train_set_index:val_set_index]
    y_val = yData[train_set_index:val_set_index]
    
    x_test = preprocessedXData[val_set_index:]
    y_test = yData[val_set_index:]
    
    return x_train, y_train, x_val, y_val, x_test, y_test
    