# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# create training data
xData, yData , times, distance = getTrainingData(6200,1400, GPM_resolution = 3)   



#%% code for the typhon qrnn
from typhon.retrieval import qrnn 
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import preprocessDataForTraining
from data_loader import load_data
import random

# load test data
#xTrain, yTrain, max1, max2 = load_data_training_data()
set_nmb = 1
newXData, newYData = load_data(set_nmb)

#newYData =np.sqrt(newYData[:,3,3])



quantiles = [0.1,0.3,0.5,0.7,0.9]
input_dim = newXData.shape[1]
#model = qrnn.QRNN(input_dim,quantiles, depth = 8,width = 256, activation = 'relu')


# split into training and validation set
if set_nmb ==1:
    cut_index_1 =175000
    cut_index_2 = len(newYData)
elif set_nmb ==2:
    cut_index_1 =160000
    cut_index_2 = len(newYData)
elif set_nmb ==3:
    cut_index_1 =3100
    cut_index_2 = len(newYData)


print(newYData.shape)
print(newXData.shape)

xTest = newXData[cut_index_1:cut_index_2,:]
yTest = newYData[cut_index_1:cut_index_2]


xTrain = newXData[:cut_index_1,:]
yTrain = newYData[:cut_index_1]
#xTrain = newXData[:cut_index_1,:]

#yTrain = newYData[:cut_index_1]


# load training data

input_dim = newXData.shape[1]


#%%

for i in range(x_mean.shape[1]):
    print(x_sigma[0,i]-model.x_sigma[0,i])
#%%

indexes = random.sample(range(0, len(xTrain)), len(xTrain))
x_val = xTrain[indexes[:20000]]
y_val = yTrain[indexes[:20000]]
x_train = xTrain[indexes[20000:]]
y_train = yTrain[indexes[20000:]]
#%%
import extendedQRNN
model = extendedQRNN.QRNN((28*28*2+4,),quantiles, depth = 8,width = 256, activation = 'relu', model_name ='CNN')

model.fit(x_train = x_train,
          y_train = y_train[:,3,3],
          x_val = x_val,
          y_val = y_val[:,3,3],
          batch_size = 512,
          maximum_epochs = 500)

#%%
print('sad')

#%% save model

model.save('CNN_model.h5')

#%% load model
from keras.models import load_model
model = load_model('results\sqrt\model.h5')
#%% load model
import extendedQRNN 
# = qrnn.QRNN()
model = extendedQRNN.QRNN.load('results\\CNN2\\model.h5')
#%%
#%%


#%%
import tensorflow as tf
print(type(yTest[0,3,3]))
print(type(prediction[0,0])) 
print(type(tf.Session().run(quantile_loss(np.float32(yTest[:,3,3]), prediction, quantiles))))
#%%
loss = tf.Session().run(quantile_loss(np.float32(yTest[:,3,3]), prediction, quantiles))
print(loss)
#%%
print(pred.shape)
#%%
import matplotlib.pyplot as plt
sum_rain = np.sum(yTest, axis = (1,2))
indexes = np.where(sum_rain > 400)[0]
#%%
print(index.shape)
#%%
index = indexes[10]

for i in range(5):
    plt.imshow(pred[index,:,:,i])
    plt.show()
plt.imshow(yTest[index,:,:])
#%%
print(np.mean(xTrain, axis=0, keepdims=True))
#%%
y_pred = model.predict(xTest)
crps = model.crps(y_pred, yTest[:,3,3], np.array(quantiles))
print(crps)
#%%
from visulize_results import generateCRPSIntervalPlot
generateCRPSIntervalPlot(crps,quantiles, yTest[:,3,3], y_pred, 1)
#%%
print(np.mean(crps))
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(newYData[:,1],newYData[:,0]))
print(correlation_target_prediction(newYData[:,0], newYData[:,1]))
#%%generate results
from visulize_results import generate_all_results
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\CNN5\\'
print(yTest.shape)
generate_all_results(model, xTest,yTest[:,3,3],yTrain[:,3,3], quantiles, False, folder_path,'Configuration 1')
#%%
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\MLP\\'
import matplotlib.pyplot as plt
prediction = np.load(folder_path+'predictions.npy')
index = 0
print(prediction[index,:])
plt.plot(prediction[index,:])
#%%
plt.plot(prediction[])
#%%
from visulize_results import generate_all_results_unet
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\u-net-100x100\\'
print(yTest.shape)
generate_all_results_unet(model, xTest,yTest,yTrain, quantiles, False, folder_path,'Configuration 6')

#%%
mean = np.zeros((30000,1))
for i in range(30000):
    
    mean[i] = model.posterior_mean(newXData[i,:])
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(mean,newYData[:,0]))
print(correlation_target_prediction(newYData[:,0], mean))
#%%
#%%
print(predictions.shape)
tmp_yTest = np.reshape(yTest, (yTest.shape[0]*yTest.shape[1]*yTest.shape[2],1))
tmp_predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1]*predictions.shape[2],predictions.shape[3]))
tmp_yTrain = np.reshape(yTrain, (yTrain.shape[0]*yTrain.shape[1]*yTrain.shape[2],1))
#%%    
print(tmp_yTest.shape)
#%%
from visulize_results import generate_all_results_CNN
generate_all_results_CNN(tmp_predictions,np.reshape(tmp_predictions[:,2],(len(tmp_predictions[:,2]),1)),None,tmp_yTest,tmp_yTrain, quantiles)
#%%

#%%
prediction = model.predict(xTest)


#%%
import matplotlib.pyplot as plt

from visulize_results import generateQQPLot
generateQQPLot(quantiles, np.reshape(np.sqrt(yTest[:,3,3]),(len(yTest),1)), prediction, title = 'Network 3')
#%%
print(prediction.shape)
#%%

from visulize_results import generate_all_results_CNN
generate_all_results_CNN(prediction,np.reshape(prediction[:,2],(len(prediction[:,2]),1)),xTest, yTest,yTrain, quantiles)
#%%
from visulize_results import generateQQPLot
generateQQPLot(quantiles, yTrain, prediction)


#%%
from data_loader import convertTimeStampToDatetime
print(convertTimeStampToDatetime(times[305000,0]))

#%%
import numpy as np
xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
#xData = xData[:,:,4:25,4:25]
'''

folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R128_P1400GPM_res3timeSeries.npy') 

'''
# remove nan values
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)

#np.mean(xTrain, axis=0, keepdims=True)

max1 = xData[:,0,:,:].max()
max2 = xData[:,1,:,:].max()
min1 = xData[:,0,:,:].min()
min2 = xData[:,1,:,:].min()
'''

max1 = np.mean(xData[:,0,:,:], axis = 0,keepdims = True)
max2 = np.mean(xData[:,1,:,:], axis = 0,keepdims = True)
min1 = np.std(xData[:,0,:,:], axis = 0,keepdims = True)
min2 = np.std(xData[:,1,:,:], axis = 0,keepdims = True)
'''
#%%
max1 = np.mean(xData[:,0,:,:])
max2 = np.mean(xData[:,1,:,:])
min1 = np.std(xData[:,0,:,:])
min2 = np.std(xData[:,1,:,:])
#%%
mean1 = np.mean(xData[:,0,:,:])
mean2 = np.mean(xData[:,1,:,:])
std1 = np.std(xData[:,0,:,:])
std2 = np.std(xData[:,1,:,:])
x_test = np.zeros((3100,112,112,2))
x_test[:,:,:,0] = (xData[:3100,0,8:120,8:120]-min1)/(max1-min1)
x_test[:,:,:,1] = (xData[:3100,1,8:120,8:120]-min2)/(max2-min2)

#%%
predictions = model.predict(x_test)

#%%
y_test = yData[:3100,:,:]
#%%
tmp = np.argwhere(yTest >30)
print(tmp.shape)
print(predictions.shape)
#%%
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
index = tmp[0,0]
fig, ax = plt.subplots(2,3, figsize= (20,6))
max_val = 50
min_val = 0.1
im1 =ax[0,0].imshow(yTest[index,:,:], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
im2 =ax[0,1].imshow(predictions[index,:,:,0],cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
im3 = ax[0,2].imshow(predictions[index,:,:,1], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
ax[1,0].imshow(predictions[index,:,:,2], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
ax[1,1].imshow(predictions[index,:,:,3], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
im6 = ax[1,2].imshow(predictions[index,:,:,4], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
cbar = fig.colorbar(im6,ax = ax, shrink = 0.79, pad = 0.025,
                   ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=20)
plt.show()
#%% load model
import matplotlib.pyplot as plt
font1 = 20
font2 = 20
fig, ax = plt.subplots(1,2, figsize= (20,6))
ax[0].hist(np.abs(times[:,0]-times[:,1]),50, color = 'black')
ax[0].set_title('Time difference (seconds)',fontsize = font2)
ax[0].set_ylabel('Frequency' ,fontsize = font1)
ax[0].set_xlabel('Abolute time difference (seconds)',fontsize = font1)
ax[0].tick_params(labelsize=15)

ax[1].hist(distance[:,1] *111,50, color = 'black')
ax[1].set_title('Distance difference (km)',fontsize = font2)
ax[1].set_ylabel('Frequency',fontsize = font1)
ax[1].set_xlabel('Distance (km)',fontsize = font1)
ax[1].tick_params(labelsize=15)

#%%
from typhon.retrieval import qrnn
# = qrnn.QRNN()
model = qrnn.QRNN.load('results/sqrt/model.h5')


#%%
from visulize_results import plot_predictions_and_labels
import datetime as datetime
DATE = datetime.datetime(2019,9,5)
plot_predictions_and_labels(model,DATE, max1,max2,min1, min2)
#%%
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
perc_tot_rate, long, lat, time = get_single_GPM_pass(DATE)





#print(time)
GPM_data = np.zeros((len(perc_tot_rate),4))
GPM_data[:,3] = perc_tot_rate
GPM_data[:,0] = long
GPM_data[:,1] = lat
GPM_data[:,2] = time
receptiveField = 28
dataSize = len(perc_tot_rate)
xData = np.zeros((dataSize,receptiveField,receptiveField,2))
times = np.zeros((dataSize,3))
yData = np.zeros((dataSize,1))
distance = np.zeros((dataSize,2))

times[:,0] = time
xData[:,:,:,0], times[:,1], distance[:,0] = getGEOData(GPM_data,'C13')
xData[:,:,:,0] = (xData[:,:,:,0]-min1)/(max1-min1)

xData[:,:,:,1], times[:,2], distance[:,1] = getGEOData(GPM_data,'C08')
xData[:,:,:,1] = (xData[:,:,:,1]-min2)/(max2-min2)

#newXData = np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]))

#newYData = np.reshape(yData,(len(yData),1))

#scaler1.fit(newXData)
#newXData = scaler1.transform(newXData)
#newYData = newYData/newYData.max()

# comine the IR images and the distance and time difference
#tmp = np.zeros((xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]+4))
#tmp[:,:xData.shape[1]*xData.shape[2]*xData.shape[3]] = newXData

'''
tmp[:,-1] = (times[:,0]-times[:,1])/(times[:,0]-times[:,1]).max()
tmp[:,-2] = distance[:,0]/distance.max()
tmp[:,-3] = (times[:,0]-times[:,2])/(times[:,0]-times[:,2]).max()
tmp[:,-4] = distance[:,1]/distance.max()
'''


#tmp[:,-1] = (times[:,0]-times[:,1])/1000
#tmp[:,-2] = distance[:,0]
#tmp[:,-3] = distance[:,1]
#tmp[:,-4] = (times[:,0]-times[:,2])/1000

#xData = xData[:,:,10:18,10:18]
#print(xData)
#print(np.mean(xData))
#print(convertTimeStampToDatetime(time[0]))
extent1  = [min(long),max(long),min(lat),max(lat)]
grid_x, grid_y = np.mgrid[extent1[0]:extent1[1]:200j, extent1[2]:extent1[3]:200j]
points = np.zeros((len(lat),2))
points[:,0] = long
points[:,1] = lat
values = perc_tot_rate
#print(grid_x)
   # print(values.max())
#upper_threshold = 20
#indexPosList = [ i for i in range(len(values)) if values[i] >upper_threshold]
#print(indexPosList)
#values[indexPosList] = upper_threshold
#print(values.max())



   

pred = model.predict(xData)
pred = np.square(pred)
max_val = max(max_val, pred.max())

#%%

extent = [min(long),max(long), min(lat)+2, max(lat)-5]
fig = plt.figure(figsize=(20, 25))
axes = []
# Generate an Cartopy projection


pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(2, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)
min_val = 0.1
max_val = max(values)
inds = np.where(values > min_val)[0]
#grid_z0 = griddata(points,values, (grid_x, grid_y), method='linear')
#im = ax.imshow(grid_z0.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
im1 = axes[-1].scatter(long[inds], lat[inds], c = values[inds], s = 1, cmap='jet',
                   norm = LogNorm(vmin=0.1, vmax = max_val)) 
axes[-1].set_title('DPR', fontsize = 14)
#plt.colorbar(im, ax=ax)

#print(pred[:,4])
#print(np.mean(pred))
# plot the precction
quantiles = [0.1,0.3,0.5,0.7,0.9]
for i in range(5):
    min_val = 0.1
    
    inds = np.where(pred[:,i] > min_val)[0]
    axes.append(fig.add_subplot(2, 3, i+2, projection=pc))
    axes[-1].set_extent(extent, crs=ccrs.PlateCarree())
    
    
    
    axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
    #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
    axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)
    axes[-1].set_title('QRNN quantile %s' % (quantiles[i]),fontsize = 14)
    
    #tmp = griddata(points,pred[:,i], (grid_x, grid_y), method='linear')
    #im =axes[-1].imshow(tmp.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
    im =axes[-1].scatter(long[inds], lat[inds], c = pred[inds,i], s = 1, cmap='jet',
                   norm = LogNorm(vmin=0.1, vmax = max_val)) 
    #plt.colorbar(im, ax=axes[-1])

cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
               ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=14)

plt.show()
    #print(np.nan_to_num(grid_z0).max())
#%%
fig = plt.figure(figsize=(10, 27))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(2, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)
quantiles = [0.1,0.3,0.5,0.7,0.9]

min_val = 0.1
max_val = max(values)
inds = np.where(values > min_val)[0]
#grid_z0 = griddata(points,values, (grid_x, grid_y), method='linear')
#im = ax.imshow(grid_z0.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
im1 = axes[-1].scatter(long[inds], lat[inds], c = values[inds], s = 1, cmap='jet',
                   norm = LogNorm(vmin=0.1, vmax = max_val)) 
axes[-1].set_title('DPR', fontsize = 14)

for i in range(5):
    min_val = 0.1
    
    inds = np.where(pred[:,i] > min_val)[0]
    axes.append(fig.add_subplot(2, 3, i+2, projection=pc))
    axes[-1].set_extent(extent, crs=ccrs.PlateCarree())
    
    
    
    axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
    #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
    axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)
    axes[-1].set_title('QRNN quantile %s' % (quantiles[i]),fontsize = 14)
    
    #tmp = griddata(points,pred[:,i], (grid_x, grid_y), method='linear')
    #im =axes[-1].imshow(tmp.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
    im =axes[-1].scatter(long[inds], lat[inds], c = pred[inds,i], s = 1, cmap='jet',
                   norm = LogNorm(vmin=0.1, vmax = max_val)) 
    #plt.colorbar(im, ax=axes[-1])

cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
               ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=14)

plt.show()
#%%
print('')


#%%
import datetime
maxLongitude = -51
minLongitde = -70
maxLatitide = 2.5
minLatitude = -11
extent = [-45,-75,7.5,-16]
 
DATE = datetime.datetime(2019,9,5)
from visulize_results import plotGPMData
plotGPMData(DATE)


#%%
import numpy as np
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S6200_R100_P1400GPM_res3interval_3.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R100_P1400GPM_res3interval_3.npy') 

#%%
print(xData.shape)
#%%
import matplotlib.pyplot as plt
index = 1
t = np.sum(yData, axis = (1,2))
print(t.shape)
tmp = np.argwhere(t >20000)
print(tmp.shape)
#%%
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
fig = plt.figure(figsize=(20, 10))

axes=[]
axes.append(fig.add_subplot(1, 3, 1))
axes.append(fig.add_subplot(1, 3, 2))
axes.append(fig.add_subplot(1, 3, 3))
#3
index = tmp[3,0]
cmin = 0.1
cmax =200
t = (xData[index,0,:,:]-xData[index,0,:,:].min())/(xData[index,0,:,:].max()-xData[index,0,:,:].min())



#im2 = axes[0].imshow(xData[index,0,:,:],cmap='jet',
#                       norm = LogNorm(vmin=0.1, vmax = 500))

im2 = axes[0].imshow(xData[index,0,:,:],norm = LogNorm(), cmap = 'cubehelix')
axes[0].set_title('Channel 13', fontsize = 17)

cbar = fig.colorbar(im2,ax = axes[0], shrink = 0.50, pad = 0.05,ticks = [0.1,0.5,1,5,10,25, 50])
cbar.ax.set_yticklabels([str(0.1),'0.5','1','5','10','25','50'])
cbar.set_label("Brightness temperature (K)", fontsize = 17)
cbar.ax.tick_params(labelsize=17)

im3 = axes[1].imshow(xData[index,1,:,:],norm = LogNorm(), cmap = 'cubehelix')
cbar = fig.colorbar(im3,ax = axes[1], shrink = 0.50, pad = 0.05,ticks = [0.1,0.5,1,2])
cbar.ax.set_yticklabels([str(0.1),'0.5','1','2'])
cbar.set_label("Brightness temperature (K)", fontsize = 17)
cbar.ax.tick_params(labelsize=17)
axes[1].set_title('Channel 8', fontsize = 17)



im1 = axes[2].imshow(yData[index,:,:] ,cmap='jet',
                       norm = LogNorm(vmin=0.1, vmax = 100))
cbar = fig.colorbar(im1,ax = axes[2], shrink = 0.50, pad = 0.05,
               ticks = [0.1,0.5,1,5,10,25, 100])
cbar.ax.set_yticklabels([str(0.1),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 17)
cbar.ax.tick_params(labelsize=17)
axes[2].set_title('2BCMB', fontsize = 17)

plt.show()
#%%
import numpy as np
nlin = 1613
ncol = 1349
filepath = "S11636382_201903200930.bin"
array = np.fromfile(filepath, dtype=np.int16,count=nlin*ncol).reshape(nlin, ncol)/10

#%%
import matplotlib.pyplot as plt
plt.imshow(array)
print(array.shape)

#%%
from os import listdir
from os.path import isfile, join
files = listdir('C:\\Users\gustav\\Documents\\Sorted\\PrecipitationMesurments\\ReferensData\\binaries')

print(files)

#%%
from data_loader import getReferenceData
getReferenceData()

#%%
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
from data_loader import preprocessDataForTraining
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S30000_R28_P10000GPM_res1reference.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S30000_R28_P10000GPM_res1reference.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S30000_R28_P10000GPM_res1reference.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S30000_R28_P10000GPM_res1reference.npy') 
GPM_data = np.load(folder_path+'/trainingData/positionC8C13S30000_R28_P10000GPM_res1reference.npy') 

import numpy as np
scalexData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
scaleyData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
scaletimes = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
scaledistance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
#xData = xData[:,:,6:22,6:22]
import extendedQRNN 
# = qrnn.QRNN()


# remove nan values
scalenanValues =np.argwhere(np.isnan(scalexData)) 
scalexData = np.delete(scalexData,np.unique(scalenanValues[:,0]),0)
#%%
from data_loader import preprocessDataForTraining
#min1 = scalexData[:,0,:,:].min()
#max1 = scalexData[:,0,:,:].max()
#min2 = scalexData[:,1,:,:].min()
#max2 = scalexData[:,1,:,:].max()
'''
model = extendedQRNN.QRNN.load('results/CNN2/model.h5')
tmpXData = np.zeros((len(xData),xData.shape[2],xData.shape[3],xData.shape[1]))
for i in range(xData.shape[1]):
 
        tmpXData[:,:,:,i] = (xData[:,i,:,:]-scalexData[:,i,:,:].min())/(scalexData[:,i,:,:].max()-scalexData[:,i,:,:].min()) 

'''


model = extendedQRNN.QRNN.load('results/MLP/model.h5')
tmpXData = np.zeros((len(xData),xData.shape[2],xData.shape[3],xData.shape[1]))
for i in range(xData.shape[1]):
        mean1 = np.mean(xData[:,i,:,:])
        std1 = np.std(xData[:,i,:,:])
        xData[:,i,:,:] =  (xData[:,i,:,:]-mean1)/std1
        

tmpXData = preprocessDataForTraining(xData, yData, times, distance)
predictions = model.predict(tmpXData)

#%%
mean = np.zeros((len(tmpXData),1))
for i in range(len(tmpXData)):
    mean[i] = model.sample_posterior(tmpXData[i,:])
#%%
from visulize_results import calculate_tot_MSE
from visulize_results import correlation_target_prediction
from visulize_results import calculate_tot_MAE
from visulize_results import calculate_bias

print(calculate_tot_MSE(mean,yData[:,0]))
print(calculate_tot_MAE(mean,yData[:,0]))
print(calculate_bias(mean,yData[:,0]))
print(correlation_target_prediction(yData[:,0], mean))
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(yData[:,1],yData[:,0]))
print(calculate_tot_MAE(yData[:,1],yData[:,0]))
print(calculate_bias(yData[:,1],yData[:,0]))
print(correlation_target_prediction(yData[:,0], yData[:,1]))
#%%
extent = [-70, -50, -10, 2]
fig = plt.figure(figsize=(30, 30))
axes = []
start_index = 20000
end_index = 30000
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)


min_val = 0.1
max_val = 100

im1 = axes[-1].scatter(GPM_data[start_index:end_index,0], GPM_data[start_index:end_index,1], c = yData[start_index:end_index,0], s = 1, cmap='jet',
               norm = LogNorm(vmin=min_val, vmax = max_val)) 
axes[-1].set_title('DPR', fontsize = 20)


axes.append(fig.add_subplot(1, 3, 2, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)

im1 = axes[-1].scatter(GPM_data[start_index:end_index,0], GPM_data[start_index:end_index,1], c = yData[start_index:end_index,1], s = 1, cmap='jet',
               norm = LogNorm(vmin=min_val, vmax = max_val)) 
axes[-1].set_title('HE', fontsize = 20)

axes.append(fig.add_subplot(1, 3, 3, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)

tmp_pred = predictions[start_index:end_index,2]
tmp_gpm = GPM_data[start_index:end_index,:]
inds = np.where(tmp_pred > min_val)[0]

im1 = axes[-1].scatter(tmp_gpm[inds,0], tmp_gpm[inds,1], c = tmp_pred[inds], s = 1, cmap='jet',
               norm = LogNorm(vmin=min_val, vmax = max_val)) 
axes[-1].set_title('QRNN', fontsize = 20)
#plt.colorbar(im, ax=ax)

   




#plt.colorbar(im, ax=axes[-1])

cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
               ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(max_val)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=20)

plt.show()
#%%
plt.plot(predictions[:,2])
#%%
from data_loader import convertTimeStampToDatetime
print(convertTimeStampToDatetime(GPM_data[0,2]))
#print(np.nan_to_num(grid_z0).max())

#%%
extent = [-70, -50, -10, 2]
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
nlin = 1613
ncol = 1349
filepath = "S11636382_201903200930.bin"
array = np.fromfile(filepath, dtype=np.int16,count=nlin*ncol).reshape(nlin, ncol)/10
# Hidro GOES16 coordinates
nlin = 1613
ncol = 1349
DY   = -0.0359477
DX   = 0.0382513
lati = 13.01202615
loni = -81.98087435


# creating lat lon vectors
latf = lati + (nlin*DY)
lonf = loni + (ncol*DX)

print(lati,latf)
print(loni,lonf)

lats = np.arange(lati,latf,DY)
lons = np.arange(loni,lonf,DX)
print(lats.shape)
print(lons.shape)
min_lat_index = np.argmin(np.abs(lats-extent[2]))
max_lat_index = np.argmin(np.abs(lats-extent[3]))
min_lon_index = np.argmin(np.abs(lons-extent[0]))
max_lon_index = np.argmin(np.abs(lons-extent[1]))
lats = lats[max_lat_index:min_lat_index]
lons = lons[min_lon_index:max_lon_index]
print(min_lat_index)
print(max_lat_index)
print(min_lon_index)
print(max_lon_index)
print(array.shape)
lats = np.reshape(lats, (len(lats),1))
lons = np.reshape(lons, (len(lons),1))
lats_matrix = np.repeat(lats,len(lons), axis=1)
lons_matrix = np.repeat(lons,len(lats), axis = 1)
print(lats_matrix.shape)
print(lats_matrix.shape)
#plt.plot(lats)
#plt.show()
#plt.plot(lons)
tmp_array = array[max_lat_index:min_lat_index, min_lon_index:max_lon_index]

fig = plt.figure(figsize=(30, 30))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)


min_val = 0.1
max_val = 100
#plt.imshow(tmp_array)
im1 = axes[-1].scatter(np.transpose(lons_matrix).flatten(),lats_matrix.flatten(), c = tmp_array.flatten(), s = 1, cmap='jet',
               norm = LogNorm(vmin=0.1, vmax = 100)) 
#%%

extent = [-70, -50, -10, 2]
fig = plt.figure(figsize=(30, 30))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)


min_val = 0.1
max_val = max(yData[:,0])

im1 = axes[-1].scatter(GPM_data[:,0], GPM_data[:,1], c = yData[:,0], s = 1, cmap='jet',
               norm = LogNorm(vmin=0.1, vmax = max_val)) 
axes[-1].set_title('DPR', fontsize = 20)

#%%


#%%
from data_loader import load_gauge_data
hydro,gauge_data, xData, times, distance = load_gauge_data()
#%%
import numpy as np
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13SreferenceTrim3Values.npy')
times = np.load(folder_path+'/trainingData/timesC8C13SreferenceTrim3Values.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13SreferenceTrim3Values.npy')
hydro = np.load(folder_path+'/trainingData/hydropredC8C13SreferenceTrim3Values.npy')
gauge_data = np.load(folder_path+'/trainingData/gaugeC8C13SreferenceTrim3Values.npy')

#%%
import matplotlib.pyplot as plt
#plt.hist(distance.flatten())
plt.hist(np.abs(times[:,1,0]-times[:,1,1]))

#%%
#%%
import numpy as np
scalexData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
scaleyData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
scaletimes = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
scaledistance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
#xData = xData[:,:,6:22,6:22]
import extendedQRNN 
# = qrnn.QRNN()

model = extendedQRNN.QRNN.load('results/MLP/model.h5')

# remove nan values
scalenanValues =np.argwhere(np.isnan(scalexData)) 
scalexData = np.delete(scalexData,np.unique(scalenanValues[:,0]),0)
predictions = np.zeros((len(gauge_data),1))

#%%
print(distance.shape)
print(times.shape)
#%%
model = extendedQRNN.QRNN.load('results/MLP/model.h5')

mean1 = np.mean(scalexData[:,0,:,:])
std1 = np.std(scalexData[:,0,:,:])
mean2 = np.mean(scalexData[:,1,:,:])
std2 = np.std(scalexData[:,1,:,:])
'''
min1 = scalexData[:,0,:,:].min()
max1 = scalexData[:,0,:,:].max()
min2 = scalexData[:,1,:,:].min()
max2 = scalexData[:,1,:,:].max()
'''
#%%
tmp_values = np.zeros((3,1))
for i in range(gauge_data.shape[0]):
    #print(gauge_data.shape)
    tmp_mean = 0
    for j in range(3):
        
        tmpXData = np.zeros((1,2,28,28))
        tmpXData[:,0,:,:] = (xData[i,j,0,:,:]-mean1)/(std1)
        tmpXData[:,1,:,:] = (xData[i,j,1,:,:]-mean2)/(std2)
        
        newXData = np.reshape(tmpXData,(1,28*28*2))
       
        tmp = np.zeros((1,28*28*2+4))
        tmp[:,:28*28*2] = newXData
        
        
        tmp[:,-1] = (times[i,j,0]-times[i,j,1])/1000
        tmp[:,-1] = 0
        tmp[:,-2] = distance[i,j,0]
        tmp[:,-3] = distance[i,j,1]
        tmp[:,-4] = (times[i,j,0]-times[i,j,2])/1000
        tmp[:,-4] = 0
        tmp_values[j,0] = model.posterior_mean(tmp)
        
        '''
        tmpXData = np.zeros((1,xData.shape[3],xData.shape[4],xData.shape[2]))
        
        tmpXData[:,:,:,0] = (xData[i,j,0,:,:]-min1)/(max1-min1)
        tmpXData[:,:,:,1] = (xData[i,j,1,:,:]-min2)/(max2-min2)
        tmp_values[j,0] = model.posterior_mean(tmpXData)
        '''
        #tmp_mean += model.posterior_mean(tmpXData)
    
    tmp_values = np.sort(tmp_values)
    predictions[i,0] = (tmp_values[0]+2*tmp_values[1]+tmp_values[2])/4
    #print(i)
#%%
print(predictions.shape)
print(gauge_data.shape)
print(hydro.shape)
print(predictions[0])
#%%
from visulize_results import calculate_tot_MSE
from visulize_results import correlation_target_prediction
from visulize_results import calculate_tot_MAE
from visulize_results import calculate_bias

print(calculate_tot_MSE(predictions,gauge_data[:,-1]))
print(calculate_tot_MAE(predictions,gauge_data[:,-1]))
print(calculate_bias(predictions,gauge_data[:,-1]))
print(correlation_target_prediction(gauge_data[:,-1], predictions))
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(hydro,gauge_data[:,-1]))
print(calculate_tot_MAE(hydro,gauge_data[:,-1]))
print(calculate_bias(hydro,gauge_data[:,-1]))
print(correlation_target_prediction(gauge_data[:,-1], hydro))

#%% scatter plot the values
import matplotlib.pyplot as plt
s=4
plt.scatter(predictions, gauge_data[:,-1], s=s, color = 'black')
plt.xlim([0,20])
plt.ylim([0,20])
plt.ylabel('Gauge value')
plt.xlabel('QRNN prediction')
plt.show()
plt.scatter(hydro, gauge_data[:,-1],s= s, color = 'black')
plt.xlim([0,20])
plt.ylim([0,20])
plt.ylabel('Gauge value')
plt.xlabel('Hydro prediction')

#%%
import numpy as np
import matplotlib.pyplot as plt
tmp_mlp = np.load('results\\MLP\\contains_true.npy')
tmp_cnn = np.load('results\\CNN2\\contains_true.npy')
tmp_t = np.load('results\\timeseries2\\contains_true.npy')
tmp_u = np.load('results\\u-net-100x100\\contains_true.npy')
plt.plot(tmp_mlp[:-1],linestyle = '--', color = 'black', label = 'Configuration 1')
plt.plot(tmp_cnn[:-1], color = 'black', label = 'Configuration 4')
#plt.plot(tmp_t[:-1],linestyle = '-.', color = 'black', label = 'Configuration 5')
#plt.plot(tmp_u[:-1],linestyle = 'dotted', color = 'black', label = 'Configuration 6')
plt.legend(fontsize = 13)
#plt.xlim([0,10])
plt.ylim([0,1])
plt.xlabel('Rain rate bins (mm/h)',fontsize = 13)
plt.ylabel('Fraction of 80% CI containing label',fontsize = 13)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
tmp_mlp = np.load('results\\MLP\\bias_label_intervals.npy')
tmp_cnn = np.load('results\\CNN2\\bias_label_intervals.npy')
tmp_t = np.load('results\\timeseries3\\bias_label_intervals.npy')
plt.plot(tmp_mlp,linestyle = '--', color = 'black', label = 'configuration 1')
plt.plot(tmp_cnn, color = 'black', label = 'configuration 4')
plt.plot(tmp_t,linestyle = '-.', color = 'black', label = 'configuration 5')
plt.legend(fontsize = 13)
#plt.xlim([0,10])
#plt.ylim([0,1])
plt.xlabel('Rain rate bins (mm/h)',fontsize = 15)
plt.ylabel('Mean bias',fontsize = 15)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
tmp_mlp = np.load('results\\MLP\\label_size_Ci_size.npy')
tmp_cnn = np.load('results\\CNN2\\label_size_Ci_size.npy')
tmp_t = np.load('results\\timeseries3\\label_size_Ci_size.npy')
plt.plot(tmp_mlp,linestyle = '--', color = 'black', label = 'configuration 1')
plt.plot(tmp_cnn, color = 'black', label = 'configuration 4')
#plt.plot(tmp_t, linestyle = '-.', color = 'black', label = 'configuration 5')

plt.legend(fontsize = 13)
#plt.xlim([0,10])
#plt.ylim([0,1])
plt.xlabel('Rain rate bins (mm/h)',fontsize = 15)
plt.ylabel('Mean CI length',fontsize = 15)

plt.show()
#%%
import matplotlib.pyplot as plt

mlp_pod = [0.246, 0.386,0.448, 0.576,0.743]
mlp_far = [0.481, 0.5,0.522,0.576,0.679]
mlp_csi = [0.201, 0.279, 0.301, 0.323, 0.289]

cnn_pod = [0.179, 0.568,0.646, 0.764, 0.873]
cnn_far = [0.327, 0.465,0.515, 0.601,0.692]
cnn_csi = [0.164,0.381,0.383, 0.355,0.294]

t_pod = [0.047,0.394, 0.561, 0.734, 0.885]
t_far =[0.232,0.385,0.49,0.607, 0.739]
t_csi = [0.047,0.316,0.365,0.344,0.252]

u_pod = [0.058, 0.526, 0.657,0.732, 0.86]
u_far = [0.254,0.43,0.515,0.581,0.696]
u_csi = [0.056,0.378,0.386,0.367,0.293]

plt.plot(cnn_pod, label = 'Configuration 4', color = 'black')
plt.plot(t_pod, label= 'Configuration 5', color = 'black', linestyle = '-.')
plt.plot(mlp_pod, label = 'Configuration 1', color = 'black', linestyle = '--')
plt.plot(u_pod, label = 'Configuration 6', color = 'black', linestyle = ':')

plt.xlabel('Quantile')
plt.ylabel('POD')
plt.xticks([0,1,2,3,4],[0.1,0.3,0.5,0.7,0.9])
plt.legend()
plt.show()

plt.plot(cnn_far, label = 'Configuration 4', color = 'black')
plt.plot(t_far, label= 'Configuration 5', color = 'black', linestyle = '-.')
plt.plot(mlp_far, label = 'Configuration 1', color = 'black', linestyle = '--')
plt.plot(u_far, label = 'Configuration 6', color = 'black', linestyle = ':')

plt.xlabel('Quantile')
plt.ylabel('FAR')
plt.xticks([0,1,2,3,4],[0.1,0.3,0.5,0.7,0.9])
plt.legend()
plt.show()

plt.plot(cnn_csi, label = 'Configuration 4', color = 'black')
plt.plot(t_csi, label= 'Configuration 5', color = 'black', linestyle = '-.')
plt.plot(mlp_csi, label = 'Configuration 1', color = 'black', linestyle = '--')
plt.plot(u_csi, label = 'Configuration 6', color = 'black', linestyle = ':')

plt.xlabel('Quantile')
plt.ylabel('CSI')
plt.xticks([0,1,2,3,4],[0.1,0.3,0.5,0.7,0.9])
plt.legend()
plt.show()

#%%
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
fig = plt.figure(figsize=(25, 15))
sum_rain = np.sum(yTest, axis = (1,2))
#sorted_1 = np.sort(sum_rain)
indexes = np.where(sum_rain > 100)[0]
index = indexes[239]
#index = sorted_1[-1]
max1 = (yTest[index,:]).max()
max2= (pred[index,:]).max()
min_val = 0.01
max_val = 100
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
#fig.tight_layout()
#fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
#fig, ax = plt.subplots()
axes=[]
plt.tight_layout()

import matplotlib.gridspec as gridspec

gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.025, hspace=0.05) #

for i in range(5):
    
    inds = np.where(pred[index,:,:,i] < min_val)
    #print(inds[0].shape)
    #print(inds[1].shape)
    axes.append(fig.add_subplot(2, 3, i+1))
    im = pred[index,:,:,i]
    im[inds[0],inds[1]] =0 
    im1 = axes[-1].imshow(im ,cmap='jet',
                       norm = LogNorm(vmin=0.1, vmax = max_val)) 
    axes[-1].set_title('%s quantile' % quantiles[i], fontsize = 20)
    axes[-1].set_aspect('equal')
#plt.colorbar(im, ax=ax)

axes.append(fig.add_subplot(2, 3, 6))
im1 = axes[-1].imshow(yTest[index,:,:] ,cmap='jet',
                   norm = LogNorm(vmin=0.1, vmax = max_val)) 
axes[-1].set_aspect('equal')
axes[-1].set_title('BCMB ', fontsize = 20)
#plt.colorbar(im, ax=ax)
cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
               ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=20)

plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
rain_values_1 = np.where(yTest[:,3,3] >0)[0]
#rain_values = np.where(yTest[:,3,3] >0)[0]
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\CNN2\\'
cnn = np.load(folder_path+'mean_predictions.npy')
for i in range(len(cnn)):
    cnn[i] = cnn[i]-yTest[i,3,3]

folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\MLP\\'
mlp = np.load(folder_path+'difference_pred_test.npy')
bins = 100
#cnn = np.abs(cnn)
#mlp = np.abs(mlp)
#plt.hist(cnn, range = [-0.0003,0.001],bins = bins, alpha = 0.4,color = 'blue',label = ['Configuration 4')
#plt.hist(mlp, range = [-0.0003,0.001],bins = bins, alpha = 0.4,color = 'blue', label = 'Configuration 1', stacked = True)
#lt.xlim([[-0.0003,0.001]])
#plt.hist(cnn,bins = bins, range = [-0.0003,0.001],alpha = 0.5,color = 'red', label = 'Configuration 4')
#plt.hist(mlp,bins = bins, range = [-0.0003,0.001],alpha = 0.5,color = 'blue', label = 'Configuration 1')
#plt.xlim([-0.0003,0.001])

print(cnn.shape)
print(yTest.shape)
#indexes_1 = np.where(np.abs(cnn) >0.002)[0]
#indexes_2 = np.where(np.abs(mlp) >0.002)[0]

plt.hist(cnn[rain_values_1],range = [-10,10],bins = bins, alpha = 0.5,color = 'red', label = 'Configuration 4')
plt.hist(mlp[rain_values_1],range = [-10,10],bins = bins, alpha = 0.5,color = 'blue', label = 'Configuration 1')
plt.xlabel('(E[y] - y_test) mm/h', fontsize = 15)
plt.ylabel('Frequancy', fontsize = 15)
plt.legend( fontsize = 13)
plt.show()

#plt.yscale('log')

#%%
import matplotlib.pyplot as plt
import numpy as np
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\CNN2\\'
cnn = np.load(folder_path+'CI_frac_lenght1.npy')

folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\MLP\\'
mlp = np.load(folder_path+'CI_frac_lenght1.npy')
sigma = 0.0001
n= 20
plt.xticks(np.arange(0, n, step = n/5),np.round(np.arange(0,n*sigma,step = sigma*n/5),4))
plt.plot(cnn, color = 'black', label = 'Configuration 4')
plt.plot(mlp, color = 'black', linestyle = '--', label = 'Configuration 1')
#tmp_x = np.arange(0, n, step = n/5)
tmp_y = np.zeros((n,1))
tmp_y[:,0] = 0.8
plt.ylim([0,1])
plt.plot(tmp_y,color = 'black', linestyle = ':')
plt.ylabel('Fraction of correct predictions',fontsize = 13)
plt.xlabel('80% Confidence interval length bins (mm/h)',fontsize = 13)

plt.legend(fontsize = 13)
plt.show()

#%%
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S6200_R100_P1400GPM_res3interval_3.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R100_P1400GPM_res3interval_3.npy') 
#%%
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
sum_rain = np.mean(yData, axis = (1,2))
plt.plot(sum_rain)
plt.show()
print(sum_rain.shape)
rain = np.where( sum_rain> 2)[0]
print(len(rain))
idx = rain[0]
fig = plt.figure(figsize=(25, 15))
min_val = 0.01
max_val = 100
axes = []
axes.append(fig.add_subplot(1,3,1))
axes[-1].imshow(xData[idx,0,:,:])
axes[-1].set_title('Channel 13 ', fontsize = 20)
axes.append(fig.add_subplot(1,3,2))
axes[-1].imshow(xData[idx,1,:,:])
axes[-1].set_title('Channel 8 ', fontsize = 20)
axes.append(fig.add_subplot(1,3,3))
inds = np.where(yData[idx,:,:] < min_val)
im = yData[idx,:,:]
im[inds[0],inds[1]] =0 
im1 = axes[-1].imshow(im ,cmap='jet',
                       norm = LogNorm(vmin=0.1, vmax = max_val))
axes[-1].set_title('Interpolated rain rates ', fontsize = 20)
#axes[-1].imshow(yData[idx,:,:])
cbar = fig.colorbar(im1,ax = axes, shrink = 0.4, pad = 0.025,
               ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=20)

#%%
import numpy as np
xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  

#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4))
axes = []
axes.append(fig.add_subplot(1,2,1))
fz = 13
fz1 = 13
axes[-1].hist(np.abs(times[:,0]-times[:,1]), bins = 100, color = 'black')
axes[-1].set_xlabel('Time difference in seconds', fontsize = fz)
axes[-1].set_ylabel('Frequency', fontsize = fz)
axes[-1].set_title('Time difference', fontsize = fz1)

axes.append(fig.add_subplot(1,2,2))
axes[-1].hist(distance[:,0], bins = 100, color = 'black')
axes[-1].set_xlabel('distance in lon lat', fontsize = fz)
axes[-1].set_ylabel('Frequency', fontsize = fz)
axes[-1].set_title('Distance difference', fontsize = fz1)
#%%

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4))
axes = []
axes.append(fig.add_subplot(1,2,1))
fz = 13
fz1 = 13
yTest = yData[:175000,3,3]
yTrain = yData[175000:,3,3]
rain_indexes = np.where(yTest > 0)[0]
axes[-1].hist(yTest[rain_indexes],range = [0,20], bins = 100, color = 'black')
axes[-1].set_xlabel('Rain rate (mm/h)', fontsize = fz)
axes[-1].set_ylabel('Frequency', fontsize = fz)
axes[-1].set_title('Test set', fontsize = fz1)

rain_indexes = np.where(yTrain > 0)[0]
axes.append(fig.add_subplot(1,2,2))
axes[-1].hist(yTrain[rain_indexes],range = [0,20], bins = 100, color = 'black')
axes[-1].set_xlabel('Rain rate (mm/h)', fontsize = fz)
axes[-1].set_ylabel('Frequency', fontsize = fz)
axes[-1].set_title('Train set', fontsize = fz1)

#%%

extent = [-90, -30, -30, 22]
extent1 = [-70, -50, -10, 2]
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm


import matplotlib.patches as patches


fig = plt.figure(figsize=(30, 30))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)

# Create a Rectangle patch
rect = patches.Rectangle((-70,-10),20,12,linewidth=1,edgecolor='black',facecolor='none')

# Add the patch to the Axes
axes[-1].add_patch(rect)
#axes[-1].set_xlabel('Degrees west')
#axes[-1].set_ylabel('Degrees north')
axes[-1].set_xticks([-70,-60, -50], crs=ccrs.PlateCarree())
axes[-1].set_yticks([-10,-4, 2], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axes[-1].xaxis.set_major_formatter(lon_formatter)
axes[-1].yaxis.set_major_formatter(lat_formatter)

plt.show()
#%%
extent = [-90, -30, -30, 22]
extent1 = [-70, -50, -10, 2]
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
DATE = datetime.datetime(2019,9,5)

import matplotlib.patches as patches
perc_tot_rate, long, lat, time = get_single_GPM_pass(DATE)


#print(time)
GPM_data = np.zeros((len(perc_tot_rate),4))
GPM_data[:,3] = perc_tot_rate
GPM_data[:,0] = long
GPM_data[:,1] = lat
GPM_data[:,2] = time
extent = [-70, -50, -10, 2]
fig = plt.figure(figsize=(15, 15))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 1, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)


min_val = 0.1
max_val = max(GPM_data[:,3])

im1 = axes[-1].scatter(GPM_data[:,0], GPM_data[:,1], c = GPM_data[:,3], s = 1, cmap='jet',
               norm = LogNorm(vmin=0.1, vmax = max_val)) 
axes[-1].set_title('2BCMB', fontsize = 20)


resolution = 100
step = (GPM_data[:,1].max()-GPM_data[:,1].min())/resolution
line_thickness = 0.3
x = np.zeros((resolution+1,1))
y = np.zeros((resolution+1,1))
x[0,0] = GPM_data[:,1].min()
idx = np.where((GPM_data[:,1] > x[0,0]-line_thickness) & (GPM_data[:,1] < x[0,0]+line_thickness) )[0]
y[0,0] = np.min(GPM_data[idx,0])
for i in range(resolution):
    x[i+1,0] = x[i] + step
    idx = np.where((GPM_data[:,1] > x[i+1,0]-line_thickness) & (GPM_data[:,1] < x[i+1,0]+line_thickness) )[0]
    
    y[i+1,0] = np.min(GPM_data[idx,0])
#print(x)
#print(y)
#axes.append(fig.add_subplot(1, 3, 2, projection=pc))
p = axes[-1].plot(y,x, color = 'black')

x[0,0] = GPM_data[:,1].min()
idx = np.where((GPM_data[:,1] > x[0,0]-line_thickness) & (GPM_data[:,1] < x[0,0]+line_thickness) )[0]
y[0,0] = np.min(GPM_data[idx,0])
for i in range(resolution):
    x[i+1,0] = x[i] + step
    idx = np.where((GPM_data[:,1] > x[i+1,0]-line_thickness) & (GPM_data[:,1] < x[i+1,0]+line_thickness) )[0]
    
    y[i+1,0] = np.max(GPM_data[idx,0])
#print(x)
#print(y)
#axes.append(fig.add_subplot(1, 3, 2, projection=pc))
axes[-1].plot(y,x, color = 'black')
axes[-1].set_xticks([-70,-60, -50], crs=ccrs.PlateCarree())
axes[-1].set_yticks([-10,-4, 2], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axes[-1].xaxis.set_major_formatter(lon_formatter)
axes[-1].yaxis.set_major_formatter(lat_formatter)
cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
               ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=15)

#plt.xlabel('tt')
#fig.xlabel('tt')

#%%
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\MLP\\'
pred1 = np.load(folder_path+'predictions.npy')
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\CNN2\\'
pred2 = np.load(folder_path+'predictions.npy')

#%%
import matplotlib.pyplot as plt
bins = 100
ran = [0,40]
plt.hist(np.abs(pred2[:,-1]-pred2[:,0]), bins = bins, range = ran, alpha = 0.5, color = 'red',label = 'Configuration 4',)
plt.hist(np.abs(pred1[:,-1]-pred1[:,0]),bins = bins , range = ran, alpha = 0.5, color = 'blue', label = 'Configuration 1')

plt.legend(fontsize = 13)
plt.xlabel('80% CI length (mm/h)', fontsize = 13)
plt.ylabel('Frequancy', fontsize = 13)
plt.yscale('log')
#%%
import numpy as np
rain_threshold = 0.0001
q = np.zeros((len(quantiles),1))
q1 = np.zeros((len(quantiles),1))
import matplotlib.pyplot as plt 
for i in range(len(quantiles)):
    nmb = 0
    nmb1 = 0
    for j in range(yTest.shape[0]):
        if pred1[j,i] > yTest[j,3,3]:
            #if yTest[j,0] == 0 and prediction[j,i] > rain_threshold:
                nmb +=1
        if pred2[j,i] > yTest[j,3,3]:
            #if yTest[j,0] == 0 and prediction[j,i] > rain_threshold:
                nmb1 +=1
    
    
    q[i,0] = nmb / yTest.shape[0]
    q1[i,0] = nmb1 / yTest.shape[0]
    
#%%
x = np.linspace(0, 1, 100)
plt.plot(quantiles,q1[:,0], color = 'black',linestyle = '-.',label = 'configuration 4')
plt.plot(quantiles,q[:,0], color = 'black',label = 'configuration 1')
plt.plot(x,x, linestyle = ':', color = 'black')
plt.legend(fontsize = 13)
plt.ylabel('Quantiles', fontsize = 13)
plt.xlabel('Observed frequency', fontsize = 13)
plt.title('Calibration plot')
plt.savefig('qq.png')
plt.show()

#%%
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\timeseries3\\'
pred1 = np.load(folder_path+'mean_values.npy')
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\u-net-100x100\\'
pred2 = np.load(folder_path+'mean_values.npy')

#%%
bins = 100
ran = [0,0.5]
plt.hist(yTest, bins = bins, range = ran, color = 'red', alpha = 0.5, label = 'Test data')
plt.hist(pred1.flatten(), bins = bins, range = ran, color = 'blue', alpha = 0.5, label = 'Configuration 5')
plt.yscale('log')
plt.legend(fontsize = 13)
plt.xlabel('Rain rate (mm/h)', fontsize = 13)
plt.ylabel('Log scaled frequancy', fontsize = 13)
#%%
import matplotlib.pyplot as plt
bins = 100
ran = [0,0.5]
plt.hist(distance[:,1]*110, bins = bins, color = 'black', label = 'Test data')
#plt.hist(pred2, bins = bins, range = ran, color = 'blue', alpha = 0.5, label = 'Configuration 4')
#plt.yscale('log')
#plt.legend(fontsize = 13)
plt.xlabel('Distance (km)', fontsize = 15)
plt.ylabel('Frequancy', fontsize = 15)
#%%
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData,extractGeoData,getIndexOfGeoDataMatricFromLongitudeLatitude
from matplotlib.colors import LogNorm
from pyproj import Proj
perc_tot_rate, long, lat, time = get_single_GPM_pass(DATE)
DATE = DATE = datetime.datetime(2019,9,5)
lons = []
lats = []
sat_h = 0
sat_lon = 0
sat_sweep = 0
x_data = []
y_data =  []
filePATH = 'ABI-L1b-RadF/2019/248/08/OR_ABI-L1b-RadF-M6C13_G16_s20192480800127_e20192480809447_c20192480809524.nc'
lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)


extent = [min(long),max(long), min(lat)+2, max(lat)-5]
fig = plt.figure(figsize=(28, 20))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)





extent1  = [min(long),max(long),min(lat),max(lat)]
grid_x, grid_y = np.mgrid[extent1[0]:extent1[1]:200j, extent1[2]:extent1[3]:200j]
points = np.zeros((len(lat),2))
points[:,0] = long
points[:,1] = lat
values = perc_tot_rate
#print(grid_x)
   # print(values.max())
#upper_threshold = 20
#indexPosList = [ i for i in range(len(values)) if values[i] >upper_threshold]
#print(indexPosList)
#values[indexPosList] = upper_threshold
#print(values.max())


min_val = 0.01
max_val = max(values)
inds = np.where(values > min_val)[0]
resolution = 100
step = (lat.max()-lat.min())/resolution
line_thickness = 0.3
x = np.zeros((resolution+1,1))
y = np.zeros((resolution+1,1))
x[0,0] = lat.min()
idx = np.where((lat > x[0,0]-line_thickness) & (lat < x[0,0]+line_thickness) )[0]
y[0,0] = np.min(lat[idx])
for i in range(resolution):
    x[i+1,0] = x[i] + step
    idx = np.where((lat > x[i+1,0]-line_thickness) & (lat < x[i+1,0]+line_thickness) )[0]
    
    y[i+1,0] = np.min(long[idx])
#print(x)
#print(y)
x1 = np.zeros((resolution+1,1))
y1 = np.zeros((resolution+1,1))
#axes.append(fig.add_subplot(1, 3, 2, projection=pc))
axes[-1].plot(y,x, color = 'black')

x1[0,0] = lat.min()
idx = np.where((lat > x1[0,0]-line_thickness) & (lat < x1[0,0]+line_thickness) )[0]
y1[0,0] = np.min(lat[idx])
for i in range(resolution):
    x1[i+1,0] = x1[i] + step
    idx = np.where((lat > x1[i+1,0]-line_thickness) & (lat < x1[i+1,0]+line_thickness) )[0]
    
    y1[i+1,0] = np.max(long[idx])
#grid_z0 = griddata(points,values, (grid_x, grid_y), method='linear')
#im = ax.imshow(grid_z0.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
axes[-1].plot(y,x, color ='black')
axes[-1].plot(y1,x1, color ='black')

im1 = axes[-1].scatter(long[inds], lat[inds], c = values[inds], s = 5, cmap='jet',
                   norm = LogNorm(vmin=0.1, vmax = max_val)) 
axes[-1].set_title('2BCMB', fontsize = 20)
cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
                   ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=14)

#print(pred[:,4])
#print(np.mean(pred))
# plot the precction

min_val = 0.1


axes.append(fig.add_subplot(1, 3, 2, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)
axes[-1].set_title('Channel 13',fontsize = 20)
axes[-1].plot(y,x, color ='black')
axes[-1].plot(y1,x1, color ='black')
#tmp = griddata(points,pred[:,i], (grid_x, grid_y), method='linear')
#im =axes[-1].imshow(tmp.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
#               norm = LogNorm(vmin=0.1, vmax = max_val)) 
#plt.colorbar(im, ax=axes[-1])
sat_h = C['goes_imager_projection'].perspective_point_height
        
# Satellite longitude
sat_lon = C['goes_imager_projection'].longitude_of_projection_origin

# Satellite sweep
sat_sweep = C['goes_imager_projection'].sweep_angle_axis
proj = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

xIndex, yIndex , distance[j,0] = getIndexOfGeoDataMatricFromLongitudeLatitude(extent[0], extent[2], proj, x_data,y_data)
diffar= +500
t_lon = lons[xIndex-diffar:xIndex, yIndex:yIndex+diffar]
t_lat = lats[xIndex-diffar:xIndex, yIndex:yIndex+diffar]
t_rad = rad[xIndex-diffar:xIndex, yIndex:yIndex+diffar]
im =axes[-1].scatter(t_lon, t_lat, c = t_rad, s = 1, cmap='cubehelix',
                       norm = LogNorm()) 



axes.append(fig.add_subplot(1, 3, 3, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)
axes[-1].set_title('Channel 8',fontsize = 20)
axes[-1].plot(y,x, color ='black')
axes[-1].plot(y1,x1, color ='black')
lons = []
lats = []
sat_h = 0
sat_lon = 0
sat_sweep = 0
x_data = []
y_data =  []
filePATH = 'ABI-L1b-RadF/2019/248/08/OR_ABI-L1b-RadF-M6C08_G16_s20192480800127_e20192480809435_c20192480809492.nc'
lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)
sat_h = C['goes_imager_projection'].perspective_point_height
        
# Satellite longitude
sat_lon = C['goes_imager_projection'].longitude_of_projection_origin

# Satellite sweep
sat_sweep = C['goes_imager_projection'].sweep_angle_axis
proj = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
xIndex, yIndex , distance[j,0] = getIndexOfGeoDataMatricFromLongitudeLatitude(extent[0], extent[2], proj, x_data,y_data)
diffar= +500
diffar= +500
t_lon = lons[xIndex-diffar:xIndex, yIndex:yIndex+diffar]
t_lat = lats[xIndex-diffar:xIndex, yIndex:yIndex+diffar]
t_rad = rad[xIndex-diffar:xIndex, yIndex:yIndex+diffar]
im =axes[-1].scatter(t_lon, t_lat, c = t_rad, s = 1, cmap='cubehelix',
                       norm = LogNorm()) 




plt.show()
#print(np.nan_to_num(grid_z0).max())

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)
#%%
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patches as patches
t = np.where(yData[:,3,3] > 5)[0]
#%%
index = t[35]

fig = plt.figure(figsize=(7, 7))
axes = []
# Generate an Cartopy projection

rect = patches.Rectangle((12.75,12.75),2.5,2.5,linewidth=1,edgecolor='r',facecolor='none')

axes.append(fig.add_subplot(1, 2, 1))
max1 = xData[:,0,:,:].max()
min1 = xData[:,0,:,:].min()
print(max1)
print(min1)
axes[-1].imshow(xData[index,0,:,:], norm = Normalize(vmin = min1,vmax=max1), cmap = 'cubehelix')
axes[-1].add_patch(rect)
axes[-1].set_title('%s' % np.round(yData[index,3,3]))

max1 = xData[:,1,:,:].max()-22
min1 = xData[:,1,:,:].min()
print(max1)
print(min1)
rect1 = patches.Rectangle((12.75,12.75),2.5,2.5,linewidth=1,edgecolor='r',facecolor='none')
axes.append(fig.add_subplot(1, 2, 2))
axes[-1].imshow(xData[index,1,:,:], norm = Normalize(vmin = min1,vmax = max1), cmap = 'cubehelix')
axes[-1].add_patch(rect1)
axes[-1].set_title('%s' % np.round(yData[index,3,3]))

#%%
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\timeseries3\\'
pred1 = np.load(folder_path+'predictions.npy')
indexes = np.where(pred1[:,4] > 10)[0]
y = [0.1, 0.3, 0.5, 0.7, 0.9]
index = indexes[0]
print(pred1[index,:])
plt.plot(pred1[index,:], y, color = 'black')
plt.xlabel('Rain rate (mm/h)')
plt.ylabel('CDF : F(y|x)')

#%%
folder_path = 'E:/Precipitation_mesurments'
import numpy as np
xData =np.load(folder_path+'/trainingData/xDataC8C13S320000_R28_P200GPM_res3timeSeries.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S320000_R28_P200GPM_res3timeSeries.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S320000_R28_P200GPM_res3timeSeries.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S320000_R28_P200GPM_res3timeSeries.npy')

nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)
#%%
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patches as patches
print(xData.shape)
t = np.where(yData > 50)[0]
#%%
index = t[2]

fig = plt.figure(figsize=(7, 7))
axes = []
# Generate an Cartopy projection

rect = patches.Rectangle((12.75,12.75),2.5,2.5,linewidth=1,edgecolor='r',facecolor='none')

axes.append(fig.add_subplot(1, 3, 1))
max1 = xData[:,0,:,:].max()
min1 = xData[:,0,:,:].min()
print(max1)
print(min1)
axes[-1].imshow(xData[index,0,:,:], norm = Normalize(vmin = min1,vmax=max1), cmap = 'cubehelix')
axes[-1].add_patch(rect)
axes[-1].set_title('%s' % np.round(yData[index]))

rect1 = patches.Rectangle((12.75,12.75),2.5,2.5,linewidth=1,edgecolor='r',facecolor='none')
axes.append(fig.add_subplot(1, 3, 2))
axes[-1].imshow(xData[index,1,:,:], norm = Normalize(vmin = min1,vmax = max1), cmap = 'cubehelix')
axes[-1].add_patch(rect1)
axes[-1].set_title('%s' % np.round(yData[index]))


rect1 = patches.Rectangle((12.75,12.75),2.5,2.5,linewidth=1,edgecolor='r',facecolor='none')
axes.append(fig.add_subplot(1, 3, 3))
axes[-1].imshow(xData[index,2,:,:], norm = Normalize(vmin = min1,vmax = max1), cmap = 'cubehelix')
axes[-1].add_patch(rect1)
axes[-1].set_title('%s' % np.round(yData[index]))


#%%
import numpy as np
import random
import matplotlib.pyplot as plt
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\MLP\\'
pred1 = np.load(folder_path+'predictions.npy')
indexes = random.sample(range(0, len(pred1)), 20000)
xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)

#%%
yTest = yData[175000:]
#%%
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\MLP\\'
pred1 = np.load(folder_path+'predictions.npy')

#%%
indexes = random.sample(range(0, len(pred1)), 20000)

for i in range(10):
    count = 0
    indexes = random.sample(range(0, len(pred1)), 20000)
    for index in indexes:
        if (pred1[index,0] < yTest[index,3,3]) & (pred1[index,-1] > yTest[index,3,3]):
            count +=1
            
    print(count / len(indexes))