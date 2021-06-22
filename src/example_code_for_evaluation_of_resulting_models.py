# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:09:16 2020

@author: gustav
This is a script showing an example of how the models can be evaluated.

"""
from visulize_results import generate_all_results,generate_all_results_unet
import numpy as np
from data_loader_preprocess import preprocess_data
import extendedQRNN 
#%%

'''
example 1: CNN results
'''

# load the model and the data

model = extendedQRNN.QRNN.load('results\\CNN2\\model.h5')

# load the data from wherever you have stored it
xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy') 

# preprocess and format the data for the MLP
model_name = 'CNN'
test_set_size = 0.5
train_set_size = 0.4
val_set_size = 0.1
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(xData, yData, model_name, train_set_size, val_set_size)
            
save = True
case = 'CNN'
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\tmp\\'

'''
if you just want to see the results formt he report you can donwload the 
files MLP_predictions.npy, MLP_mean_values.npy, CNN_predictions.npy and 
CNN_mean_values.npy and place them at your folder_path with the names
predictions.npy and mean_values.npy. See the code for generate_all_results() for 
more detail.
'''
generate_all_results(model,x_test, y_test[:,3,3], y_train ,quantiles, save, folder_path, case)

#%%
'''
example 2: MLP results
'''

# load the model and the data
model = extendedQRNN.QRNN.load('results\\adriano\\model_mlp.h5')

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

save = True 
case = 'MLP'
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\adriano\\'

generate_all_results(model,x_test, y_test[:,3,3], y_train ,quantiles, save, folder_path, case)

#%%
'''
example 2: Unet results
'''

# load the model and the data
model = extendedQRNN.QRNN.load('results\\u-net-100x100\\model.h5')

# load the data from wherever you have stored it
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S6200_R100_P1400GPM_res3interval_3.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R100_P1400GPM_res3interval_3.npy') 

# preprocess and format the data for the MLP
model_name = 'unet'
test_set_size = 0.5
train_set_size = 0.4
val_set_size = 0.1
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(xData, yData,times,distance, model_name, train_set_size, val_set_size)

save = True 
case = 'unet'
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\unetAdriano\\'

generate_all_results_unet(model,x_test, y_test, y_train ,quantiles, save, folder_path, case)

