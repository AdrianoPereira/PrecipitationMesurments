# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:10:20 2020

@author: gustav
"""
no_rain_value = 0.01
def generateQQPLot(quantiles, yTest, prediction, case):
    import numpy as np
    rain_threshold = 0.0001
    q = np.zeros((len(quantiles),1))
    import matplotlib.pyplot as plt 
    
    for i in range(len(quantiles)):
        nmb = 0
        for j in range(yTest.shape[0]):
            if prediction[j,i] > yTest[j]:
                #if yTest[j,0] == 0 and prediction[j,i] > rain_threshold:
                    nmb +=1
        
        
        q[i,0] = nmb / yTest.shape[0]
        
    
    x = np.linspace(0, 1, 100)
    plt.plot(quantiles,q[:,0], color = 'black')
    plt.plot(x,x, linestyle = ':', color = 'black')
    plt.ylabel('Quantiles')
    plt.xlabel('Observed frequency')
    plt.title(case)
    plt.savefig('qq.png')
    plt.show()
def plot_small_pred_histograms(mean, yTest, case):
    import numpy as np
   
    import matplotlib.pyplot as plt 
    
    indexes = np.where(yTest >0)[0]
    plt.hist(yTest[indexes], bins = 50, range = [0,0.2], color = 'black')
    #plt.plot(x,x, linestyle = ':', color = 'black')
    plt.ylabel('Frequancy')
    plt.xlabel('rain rate mm/h')
    plt.title('Dataset 1 test data')
    plt.show()
    indexes = np.where(mean >0.01)[0]
    plt.hist(mean[indexes], bins = 50, range = [0,0.2], color = 'black')
    plt.ylabel('Frequancy')
    plt.xlabel('rain rate mm/h')
    plt.title(case)
    plt.show()
def scatterplot(mean, yTest, case):
    import matplotlib.pyplot as plt
    plt.scatter(mean, yTest, s=0.05,alpha = 1, color = 'black')
    plt.xlim([0,2])
    plt.ylim([0,2])
    plt.ylabel('Label')
    plt.xlabel('Prediction')
    plt.title(case)
    plt.show()
 
def histogram_plot(mean, yTest, case, folder_path, pred_range):
    import matplotlib.pyplot as plt
    import numpy as np
    result = np.zeros((len(yTest),1))
    for i in range(len(yTest)):
       result[i] = mean[i]-yTest[i]
        
    plt.hist(yTest, range = pred_range,bins = 200,alpha = 0.5, color = 'red', label = 'Test set')
    plt.hist(mean,  range = pred_range,bins = 200, alpha = 0.5,color = 'blue',label = 'Expected value predictions')
    plt.xlabel('rain rate (mm/h)')
    plt.ylabel('Log scaled frequency')
    
    #plt.ylim([0,600])
    #plt.hist(result, range = [-50,20],bins = 50, alpha = 0.3,color = 'blue')
    np.save(folder_path+'difference_pred_test.npy', result)
    
    plt.yscale('log')
    plt.legend()
    plt.show()
    

def scatterplot_zoom(mean, yTest, case):
    import matplotlib.pyplot as plt
    plt.scatter(mean, yTest, s=5,alpha = 0.3, color = 'black')
    plt.xlim([0,0.2])
    plt.ylim([0,0.2])
    plt.ylabel('Label')
    plt.xlabel('Prediction')
    plt.title(case)
    plt.show()
    
def CISizeDistribution(predictions,case):
    import numpy as np
    sigma = 0.005
    import matplotlib.pyplot as plt
    
    pred_sizes = np.abs(predictions[:,0]-predictions[:,-1])
    #indexes = np.where((pred_sizes < sigma*(k+1)) & (pred_sizes >= 0))[0]
    plt.hist(pred_sizes,bins = 100, range=[0,0.01], color = 'black')
    #plt.xlim([0,0.005])
    #print(np.mean(pred_sizes))
    #print(np.median(pred_sizes))
    #indexx = np.argmax(pred_sizes)
    #print(predictions[indexx,:])
    #plt.plot(pred_sizes)
    #plt.show()
    '''
    results = np.zeros((5,1))
    length = len(predictions)
    
    for k in range(5):
        indexes = np.where((pred_sizes < sigma*(k+1)) & (pred_sizes >= sigma*(k)))[0]
        
        results[k,0] = len(indexes)/length
        
    plt.plot(results, color = 'black')
    '''
    plt.ylabel('Number of occurencies')
    plt.xlabel('80% Confidence interval length')
    #plt.title('')
    plt.title(case)
    plt.show()
def getCIsizeFractionPlot(predictions, yTest,case, sigma, n,folder_path):
    import numpy as np
    #sigma = 0.0001
   # n=80
    import matplotlib.pyplot as plt
    pred_sizes = np.abs(predictions[:,0]-predictions[:,-1])
    results = np.zeros((n,1))
    
    for k in range(n):
        indexes = np.where((pred_sizes < sigma*(k+1)) &(pred_sizes >= sigma*(k)))[0]
        #print(len(indexes))
        #print(k)
        count = 0
        for index in indexes:
            if (predictions[index,0] < yTest[index]) & (predictions[index,-1] > yTest[index]):
                count+=1
        if len(indexes) >0:
            results[k,0] = count/len(indexes)
        else:
            results[k,0] = 0
    plt.xticks(np.arange(0, n, step = n/5),np.round(np.arange(0,n*sigma,step = sigma*n/5),4))
    plt.plot(results, color = 'black')
    #tmp_x = np.arange(0, n, step = n/5)
    tmp_y = np.zeros((n,1))
    tmp_y[:,0] = 0.8
    plt.ylim([0,1])
    plt.plot(tmp_y,color = 'black', linestyle = '--')
    plt.ylabel('fraction of correct prediction')
    plt.xlabel('80% Confidence interval length bins')
    plt.title(case)
    plt.show()
    np.save(folder_path+'CI_frac_lenght2.npy', results)
def getBiasPerIntervals(mean, yTest, sigma, folder_path):
    import numpy as np
    import matplotlib.pyplot as plt  
    bias_intervals = np.zeros((11,1))
    indexes = np.where(yTest == 0)[0]
    bias_intervals[0] = calculate_bias(mean[indexes], yTest[indexes])
    
    for k in range(10):
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        bias_intervals[k+1] = calculate_bias(mean[indexes], yTest[indexes])
    plt.plot(bias_intervals, color = 'black')
    
    plt.ylabel('bias')
    plt.xlabel('interval number')
    plt.title('Bias for rain rate bins')
    np.save(folder_path+'bias_label_intervals.npy', bias_intervals)
    plt.show()
    
def pltCRPSPerIntervalSize(crps,quantiles, yTest, predictions, case, sigma, n):
    import numpy as np
    import matplotlib.pyplot as plt
    
    #sigma = 0.05
    #n=200
    pred_sizes = np.abs(predictions[:,0]-predictions[:,-1])
    results = np.zeros((n,1))
    
    for k in range(n):
        indexes = np.where((pred_sizes < sigma*(k+1)) &(pred_sizes >= sigma*(k)))[0]
        #print(len(indexes))
        #print(k)
        results[k] = np.mean(crps[indexes])
        
    plt.xticks(np.arange(0, n, step = n/5),np.round(np.arange(0,n*sigma,step = sigma*n/5),3))
    plt.plot(results, color = 'black')
    #tmp_x = np.arange(0, n, step = n/5)
    #tmp_y = np.zeros((n,1))
    #tmp_y[:,0] = 0.8
    #plt.ylim([0,1])
    #plt.plot(tmp_y,color = 'black', linestyle = '--')
    plt.ylabel('mean CRPS')
    plt.xlabel('80% Confidence interval length bins')
    plt.title(case)
    plt.show()
def loss(yTest, prediction):
    import tensorflow as tf
    import numpy as np
    quantiles  = [0.1,0.3,0.5,0.7,0.9]
    return tf.Session().run(quantile_loss(np.float32(yTest), prediction, quantiles))
def pltLossPerIntervalSize(quantiles, yTest, predictions, case, sigma, n):
    import numpy as np
    import matplotlib.pyplot as plt
    
    #sigma = 0.05
    #n=200
    pred_sizes = np.abs(predictions[:,0]-predictions[:,-1])
    results = np.zeros((n,1))
    
    for k in range(n):
        indexes = np.where((pred_sizes < sigma*(k+1)) &(pred_sizes >= sigma*(k)))[0]
        #print(len(indexes))
        #print(k)
        results[k] = np.mean(loss(yTest[indexes],predictions[indexes]))
        
    plt.xticks(np.arange(0, n, step = n/5),np.round(np.arange(0,n*sigma,step = sigma*n/5),3))
    plt.plot(results, color = 'black')
    #tmp_x = np.arange(0, n, step = n/5)
    #tmp_y = np.zeros((n,1))
    #tmp_y[:,0] = 0.8
    #plt.ylim([0,1])
    #plt.plot(tmp_y,color = 'black', linestyle = '--')
    plt.ylabel('mean Loss')
    plt.xlabel('80% Confidence interval length bins')
    plt.title(case)
    plt.show()
def plotLabelSizeVsCISize(predictions, yTest, sigma, case,folder_path):
    import numpy as np
    import matplotlib.pyplot as plt  
    CI_sizes = np.zeros((11,1))
    indexes = np.where(yTest == 0)[0]
    CI_sizes[0] = np.mean(predictions[indexes,-1]-predictions[indexes,0])
    
    for k in range(10):
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        print(len(indexes))
        CI_sizes[k+1] = np.mean(predictions[indexes,-1]-predictions[indexes,0])
    plt.plot(CI_sizes, color = 'black')
    
    plt.ylabel('mean CI length')
    plt.xlabel('label size')
    plt.title(case)
    plt.show()
    np.save(folder_path+'label_size_Ci_size.npy', CI_sizes)
def generate_qqplot_for_intervals(quantiles, yTest, prediction, sigma):
    import numpy as np
    import matplotlib.pyplot as plt  
    fig, ax=plt.subplots(nrows=10, ncols=1,figsize=(5,50))
    for k in range(10):
       
        # select the indexes within the specific interval
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        tmp_y = yTest[indexes]
        tmp_pred = prediction[indexes,:]
        
        q = np.zeros((len(quantiles),1))
        for i in range(len(quantiles)):
            nmb = 0
            for j in range(tmp_y.shape[0]):
                if tmp_pred[j,i] > tmp_y[j]:
                    nmb +=1
            
            if tmp_y.shape[0] != 0:
                
                q[i,0] = nmb / tmp_y.shape[0]
            else:
                q[i,0] = 0
            
      
        x = np.linspace(0, 1, 100)
        ax[k].set_title('interval: ('+str(sigma*k)+','+str(sigma*(k+1))+')'+' amount of observations'+ str(len(indexes)))
        ax[k].plot(quantiles,q[:,0])
        ax[k].plot(x,x)
    
    plt.show()
    plt.savefig('qq_intervals.png')
def generateCRPSIntervalPlot(crps,quantiles, yTest, prediction, sigma):
    import numpy as np
    import matplotlib.pyplot as plt  
    crps_intervals = np.zeros((11,1))
    indexes = np.where(yTest ==0 )[0]
    crps_intervals[0] = np.mean(crps[indexes])
    for k in range(10):
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        crps_intervals[k+1] = np.mean(crps[indexes])
    plt.plot(crps_intervals, color = 'black')
    plt.ylabel('Mean CRPS')
    plt.xlabel('interval number')
    plt.title('Mean CRPS for intervals')
    plt.show()
    
def containsTrueValueIntervals(quantiles, yTest, prediction, sigma, folder_path,case):
    import numpy as np
    import matplotlib.pyplot as plt  
    result = np.zeros((11,1))
    indexes = np.where(yTest ==0)[0]
    correct = 0
    # no rain case
    for index in indexes:
        if yTest[index]<prediction[index,-1] and yTest[index] > prediction[index,0]:
            correct+=1
    
    result[0,0] = correct / len(indexes)
    
    for k in range(10):
        
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        correct = 0
        for index in indexes:
            if yTest[index]<prediction[index,-1] and yTest[index] > prediction[index,0]:
                correct+=1
        
        result[k,0] = correct / len(indexes)
        
    
    plt.plot(result, color = 'black')
    plt.ylabel('Procentage containing true value')
    plt.xlabel('Interval number')
    plt.title(case)
    plt.show()
    np.save(folder_path+'contains_true.npy', result)
        
def getMeansSquareError(yTest, mean, sigma):
    import numpy as np
    import matplotlib.pyplot as plt  
    fig, ax=plt.subplots()
    error = np.zeros((16))
    # select the indexes within the specific interval
    indexes = np.where(yTest == 0)[0]
    tmp_y = yTest[indexes]
    tmp_pred = mean[indexes]
    t_sum = 0
    for i in range(len(tmp_y)):
        t_sum += np.abs(tmp_y[i]-tmp_pred[i])
    error[0] = t_sum/len(tmp_y)
    for k in range(15):
       
        # select the indexes within the specific interval
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        tmp_y = yTest[indexes]
        tmp_pred = mean[indexes]
        
        #error[k+1] = (np.abs(tmp_y-tmp_pred)*np.abs(tmp_y-tmp_pred)).sum()/len(tmp_y)
        error[k+1] = calculate_tot_MSE(mean[indexes], yTest[indexes])
     
    ax.plot(error)
    plt.show()
    #plt.savefig('MSE_interval.png')
    
def plotIntervalPredictions(yTest, prediction, sigma):
    import numpy as np
    import matplotlib.pyplot as plt  
    fig, ax=plt.subplots(nrows=10, ncols=1,figsize=(5,50))
    for k in range(10):
       
        # select the indexes within the specific interval
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        tmp_y = yTest[indexes]
        tmp_pred = prediction[indexes,:]
    
            
      
        ax[k].set_title('mean pred:'+ str(np.mean(tmp_pred))+'mean actual:'+str(np.mean(tmp_y)))
        ax[k].plot(tmp_y)
        ax[k].plot(tmp_pred,alpha = 0.5)
    
    plt.show()
def confusionMatrix(yTest, predictions):
    #conf_matrix = np.zeros(2)
    
    pred_rain_was_rain = 0
    pred_rain_wasnt_rain= 0
    pred_no_rain_was_rain = 0
    pred_no_rain_wasnt_rain = 0
    #print(yTest.shape)
    #print(predictions.shape)
    
    for i in range(len(yTest)):
        
        if yTest[i] <= 0 and predictions[i] <= 0:
            pred_no_rain_wasnt_rain +=1
        
        elif yTest[i] <= 0 and predictions[i] > 0:
            pred_rain_wasnt_rain +=1
        
        elif yTest[i] > 0 and predictions[i] <= 0:
            pred_no_rain_was_rain +=1
        
        elif yTest[i] > 0 and predictions[i] > 0:
            pred_rain_was_rain +=1 
    
    print('pred_rain_was_rain:'+str(pred_rain_was_rain)+'pred_rain_wasnt_rain:'+str(pred_rain_wasnt_rain)+'pred_no_rain_was_rain:'+str(pred_no_rain_was_rain)+'pred_no_rain_wasnt_rain:'+str(pred_no_rain_wasnt_rain))
def generateRainfallImage(model,data_file_paths):
    from data_loader import downloadFile
    from data_loader import extractGeoData
    import numpy as np
    import matplotlib.pyplot as plt
    
    im_width = 200
    field_of_vision = 8
    half_field_of_vision = 4
    test = np.zeros((im_width*im_width,len(data_file_paths),field_of_vision,field_of_vision))
    channel =0
    
    for file in data_file_paths:
        #downloadFile(file)
        lons,lats,C,rad, x_data, y_data = extractGeoData(file)
        rad = rad.data
        
        rad = rad[:im_width,:im_width]
       
       
        print(rad.shape)
        
        # generate rad images to be evaluated
        index = 0
        for i in range(rad.shape[0]):
            #print(i)
            for j in range(rad.shape[1]):
                #print(np.reshape(rad[i-3:i+3,j-3:j+3].data,(1,36))/max_val)
                if i<half_field_of_vision or i >rad.shape[0]-half_field_of_vision or j<half_field_of_vision or j >rad.shape[1]-half_field_of_vision :
                    test[index,channel,:,:] = rad[:field_of_vision,:field_of_vision]
                else:
                    test[index,channel,:,:] = rad[i-half_field_of_vision:i+half_field_of_vision,j-half_field_of_vision:j+half_field_of_vision]
                #predictions[i,j] = model.predict(np.reshape(rad[i-3:i+3,j-3:j+3],(1,36))/max_val)[0,4]
                index = index +1
        channel +=1
            
    test2 = np.reshape(test,(rad.shape[0]*rad.shape[1],len(data_file_paths)*field_of_vision*field_of_vision))
    test3 = np.zeros((rad.shape[0]*rad.shape[1],len(data_file_paths)*field_of_vision*field_of_vision+4))
    test3[:,:len(data_file_paths)*field_of_vision*field_of_vision] = test2
    pred = model.predict(test3)
    #fig, axes = plt.subplots(nrows=2, ncols=3)
    plt.figure(figsize=(20, 20))
    
    for i in range(pred.shape[1]):
        
        ax = plt.subplot(3, 3, i+1)
        
        #fig, ax = plt.subplots()
        img = pred[:,i].reshape(im_width,im_width)
        #img_in =  np.ma.masked_less(img, 0.001)
        #im = axes.flat[i].imshow(img,vmin=0, vmax=20)
        #im = axes.flat[i].imshow(img)
        
        #fig.colorbar(im)
        im =plt.imshow(img)
        #plt.title(i)
        #im = ax.imshow(img_in)
        plt.colorbar(im, ax=ax)
    
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plotGPMData(DATE):
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
    extent = [-70, -50, -12, 5]
    fig = plt.figure(figsize=(20, 20))


    # Generate an Cartopy projection
    pc = ccrs.PlateCarree()
    #fig.tight_layout()
    #fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
    #fig, ax = plt.subplots()
    axes=[]
    axes.append(fig.add_subplot(1, 1, 1, projection=pc))
    axes[-1].set_extent(extent, crs=ccrs.PlateCarree())
    
    
    
    axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
    #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
    axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)
    
    
    
    
    perc_tot_rate, long, lat, time = get_single_GPM_pass(DATE)
    #print(time)
    GPM_data = np.zeros((len(perc_tot_rate),4))
    GPM_data[:,3] = perc_tot_rate
    GPM_data[:,0] = long
    GPM_data[:,1] = lat
    GPM_data[:,2] = time
    
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
    
    
    min_val = 0.1
    max_val = max(values)
    inds = np.where(values > min_val)[0]
    #grid_z0 = griddata(points,values, (grid_x, grid_y), method='linear')
    #im = ax.imshow(grid_z0.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
    im1 = axes[-1].scatter(long[inds], lat[inds], c = values[inds], s = 1, cmap='jet',
                       norm = LogNorm(vmin=0.1, vmax = max_val)) 
    axes[-1].set_title('DPR', fontsize = 20)
    #plt.colorbar(im, ax=ax)
    
    
    cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
                   ticks = [min_val,0.5,1,5,10,25, max_val])
    cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
    cbar.set_label("Rain rate (mm/h)", fontsize = 20)
    cbar.ax.tick_params(labelsize=20)
    
    plt.show()
    #print(np.nan_to_num(grid_z0).max())
    
def plotGOESData(FILE, extent):
    from datetime import datetime
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import metpy  # noqa: F401
    import numpy as np
    import xarray
    from pyproj import Proj
    '''
    maxLongitude = -51
    minLongitde = -70
    maxLatitide = 2.5
    minLatitude = -11
    '''
    maxLongitude = -40
    minLongitde = -80
    maxLatitide = 8
    minLatitude = -20

    # Open the file with xarray.
    # The opened file is assigned to "C" for the CONUS domain.
    
    #FILE = ('data/OR_ABI-L1b-RadF-M6C13_G16_s20200260150156_e20200260159476_c20200260159547.nc')
    C = xarray.open_dataset(FILE)
     # Satellite height
    sat_h = C['goes_imager_projection'].perspective_point_height
    
    # Satellite longitude
    sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
    
    # Satellite sweep
    sat_sweep = C['goes_imager_projection'].sweep_angle_axis
    
    # The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
    # See details here: https://proj4.org/operations/projections/geos.html?highlight=geostationary
    x_new = C['x'][:] * sat_h
    y_new = C['y'][:] * sat_h
    #x = C['x'][:] 
    #y = C['y'][:] 
    rad = C['Rad']
    #Dataset('data/OR_ABI-L1b-RadC-M6C01_G16_s20200160001163_e20200160003536_c20200160004010.nc','r')['nominal_satellite_subpoint_lat'][:]
    # Create a pyproj geostationary map object
    
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
  

    xmin_proj, ymax_proj = p(minLongitde, maxLatitide)
    xmax_proj, ymin_proj = p(maxLongitude, minLatitude)
    
    print()
    xmin_proj =min([xmin_proj,xmax_proj])
    xmax_proj =max([xmin_proj,xmax_proj])
    ymin_proj =min([ymin_proj,ymax_proj])
    ymax_proj =max([ymin_proj,ymax_proj])
    
    print(xmin_proj)
    print(xmax_proj)
    print(ymin_proj)
    print(ymax_proj)
    
    
    xmin_index = (np.abs(x_new.data-xmin_proj)).argmin()
    xmax_index = (np.abs(x_new.data-xmax_proj)).argmin()
    ymin_index = (np.abs(y_new.data-ymin_proj)).argmin()
    ymax_index = (np.abs(y_new.data-ymax_proj)).argmin()
     
    print(xmin_index)
    print(xmax_index)
    print(ymin_index)
    print(ymax_index)

    
    x_new = x_new[xmin_index:xmax_index]
    y_new = y_new[ymax_index:ymin_index]
    print(x_new)
    print(y_new)
    print(x_new.coords['x'])
    x_new.coords['x'] = x_new.coords['x']* sat_h
    y_new.coords['y'] = y_new.coords['y']* sat_h
    print(x_new)
    rad = rad[ymax_index:ymin_index,xmin_index:xmax_index]
    rad.coords['x'] =rad.coords['x']*sat_h
    rad.coords['y'] =rad.coords['y']*sat_h
    print(rad)
    #rad = rad[xmin_index.data:xmax_index.data,ymax_index.data:ymin_index.data]
    # We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
    dat = C.metpy.parse_cf('Rad')
    
    geos = dat.metpy.cartopy_crs
    
    x = dat.metpy.x
    y = dat.metpy.y
    print(x)
    print(y)
    #long = dat.metpy.longitude
    #lat = dat.metpy.latitude
    #print(long)
    x = x_new
    y = y_new
    
    fig = plt.figure(figsize=(15, 12))
    # Generate an Cartopy projection
    pc = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=pc)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    
    ax.imshow(rad, origin='upper',
              extent=(x.min(), x.max(), y.min(), y.max()),
              transform=geos,
              interpolation='none')
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    
    #plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
    #plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')
    
    plt.show()

def plot_predictions_and_labels(model,DATE, max1, max2,min1, min2):
    
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
    #xData[:,:,:,0] = (xData[:,:,:,0]-max1)/(min1)
    xData[:,:,:,1], times[:,2], distance[:,1] = getGEOData(GPM_data,'C08')
    xData[:,:,:,1] = (xData[:,:,:,1]-min2)/(max2-min2)
    #xData[:,:,:,1] = (xData[:,:,:,1]-max2)/(min2)
    
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
    
    im1 = axes[-1].scatter(long[inds], lat[inds], c = values[inds], s = 1, cmap='jet',
                       norm = LogNorm(vmin=0.1, vmax = max_val)) 
    axes[-1].set_title('2BCMB', fontsize = 14)
    
    #plt.colorbar(im, ax=ax)
    print(x)
    print(y)
    print(x1)
    print(y1)
   
    #print(x)
    #print(y)
    #axes.append(fig.add_subplot(1, 3, 2, projection=pc))
    
    
    pred = model.predict(xData)
    #pred = np.square(pred)
    max_val = max(max_val, pred.max())
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
        axes[-1].plot(y,x, color ='black')
        axes[-1].plot(y1,x1, color ='black')
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
    
def calculate_POD(predictions, targets):
    
    '''
    returns the probability of detection
    
    '''
    
    
    import numpy as np
    rain_indexes = np.where(targets >no_rain_value )[0]
    true_possitives =0
    missing_values = 0
    for rain_index in rain_indexes:
        
        if predictions[rain_index] > no_rain_value:
            true_possitives +=1
        else:
            missing_values +=1
            
    return np.round(true_possitives/(true_possitives+missing_values),3)

def calculate_FAR(predictions, targets):
    
    '''
    returns the false alarm ratio
    '''
    
    import numpy as np
    predicted_rain_indexes = np.where(predictions > no_rain_value)[0]
    true_possitives = 0
    false_pssitives = 0
    
    for rain_index in predicted_rain_indexes:
        if targets[rain_index] > no_rain_value:
            true_possitives +=1
        else:
            false_pssitives +=1
    return np.round(false_pssitives /(false_pssitives+true_possitives),3)

def calculate_CSI(predictions, targets):
    import numpy as np
    '''
    returns the critical sucess index
    '''
    
    
    rain_indexes = np.where(targets > 0)[0]
    predicted_rain_indexes = np.where(predictions > no_rain_value)[0]
    true_possitives =0
    missing_values = 0
    false_pssitives = 0
    
    for rain_index in rain_indexes:
        
        if predictions[rain_index] > no_rain_value:
            true_possitives +=1
        else:
            missing_values +=1
            
    for rain_index in predicted_rain_indexes:
        if targets[rain_index] == 0:
            false_pssitives +=1
            
    return np.round(true_possitives/(true_possitives+false_pssitives+missing_values),3)

def calculate_bias(predictions, targets):
    return predictions.mean()-targets.mean()

def calculate_tot_MSE(predictions,targets):
    
    tot =0
    for i in range(len(predictions)):
        tot += (predictions[i]-targets[i])*(predictions[i]-targets[i])
    return tot/len(predictions)
def calculate_tot_MAE(predictions,targets):
    import numpy as np
    tot =0
    for i in range(len(predictions)):
        tot += np.abs(predictions[i]-targets[i])
    return tot/len(predictions)

def get_apriori_mean_estimate_kvote(y_test,y_train, mean):
    import numpy as np
    a_priori_mean = np.mean(y_train)
    MSE_a_priori_mean =0
    MSE_mean = 0
    
    for i in range(len(y_test)):
        MSE_a_priori_mean += (a_priori_mean-y_test[i])*(a_priori_mean-y_test[i])
        MSE_mean += (mean[i]-y_test[i])*(mean[i]-y_test[i])
    
    return MSE_mean/MSE_a_priori_mean

def get_apriori_median_estimate_kvote(y_test,y_train, mean):
    import numpy as np
    a_priori_mean = np.mean(y_train)
    MSE_a_priori_mean =np.zeros((len(y_test),1))
    MSE_mean = np.zeros((len(y_test),1))
    
    for i in range(len(y_test)):
        MSE_a_priori_mean[i] = (a_priori_mean-y_test[i])*(a_priori_mean-y_test[i])
        MSE_mean[i] = (mean[i]-y_test[i])*(mean[i]-y_test[i])
    
    return np.median(MSE_mean)/np.median(MSE_a_priori_mean)

def get_apriori_mean_estimate_kvote_abs_mean(y_test,y_train, mean):
    import numpy as np
    a_priori_mean = np.mean(y_train)
    MSE_a_priori_mean =0
    MSE_mean = 0
    for i in range(len(y_test)):
        MSE_a_priori_mean += np.abs((a_priori_mean-y_test[i]))
        MSE_mean += np.abs((mean[i]-y_test[i]))
    
    return MSE_mean/MSE_a_priori_mean

def get_apriori_median_estimate_kvote_abs_mean(y_test,y_train, mean):
    import numpy as np
    a_priori_mean = np.mean(y_train)
    MSE_a_priori_mean = np.zeros((len(y_test),1))
    MSE_mean = np.zeros((len(y_test),1))
    for i in range(len(y_test)):
        MSE_a_priori_mean[i] = np.abs((a_priori_mean-y_test[i]))
        MSE_mean[i] = np.abs((mean[i]-y_test[i]))
    
    return np.median(MSE_mean)/np.median(MSE_a_priori_mean)

def get_correlation_labelsize_prediction(yTest, prediction):
    import numpy as np
    exy = 0
    ex = 0
    #ey = 0
    for i in range(len(prediction)):
       if yTest[i] > prediction[i,0] and yTest[i] < prediction[i,4]:
           ex +=1
           exy += yTest[i]
       
    
    return (exy/len(prediction)-ex/len(prediction)*yTest.sum()/len(prediction))

def get_correlation_MSE_labelsize(yTest, prediction):
    import numpy as np
    exy = 0
    ex = 0
    varx1 = 0
    #varx2 = 0
    #ey = 0
    for i in range(len(prediction)):
        
        varx1 += np.abs(yTest[i]-prediction[i,2])*np.abs(yTest[i]-prediction[i,2])
        ex += np.abs(yTest[i]-prediction[i,2])
        exy += yTest[i]*np.abs(yTest[i]-prediction[i,2])
    
    var = varx1/len(prediction)-(ex/len(prediction))*(ex/len(prediction))
    
    return (exy/len(prediction)-ex/len(prediction)*yTest.sum()/len(prediction))/(np.std(yTest)*np.sqrt(var))
def get_correlation_MSEapriori_labelsize(yTest,yTrain, prediction):
    import numpy as np
    exy = 0
    ex = 0
    varx1 = 0
    #varx2 = 0
    #ey = 0
    pred = np.mean(yTrain)
    for i in range(len(prediction)):
        
        varx1 += np.abs(yTest[i]-pred)*np.abs(yTest[i]-pred)
        ex += np.abs(yTest[i]-pred)
        exy += yTest[i]*np.abs(yTest[i]-pred)
    
    var = varx1/len(prediction)-(ex/len(prediction))*(ex/len(prediction))
    
    return (exy/len(prediction)-ex/len(prediction)*yTest.sum()/len(prediction))/(np.std(yTest)*np.sqrt(var))
def correlation_target_prediction(yTest, prediction):
    import numpy as np
    exy = 0
    ex = 0
    #ey = 0
    for i in range(len(prediction)):
      
        ex += prediction[i]
        exy += yTest[i]*prediction[i]
    
        #ex += np.abs(yTest[i]-prediction[i,2])
        #exy += yTest[i]*np.abs(yTest[i]-prediction[i,2])
    
    
    return (exy/len(prediction)-ex/len(prediction)*yTest.sum()/len(prediction))/(np.std(yTest)*np.std(prediction[:]))

def cdf(prediction, quantiles):
        import numpy as np
        r"""
        Approximate the posterior CDF for given inputs `x`.
        Propagates the inputs in `x` forward through the network and
        approximates the posterior CDF by a piecewise linear function.
        The piecewise linear function is given by its values at
        approximate quantiles $x_\tau$ for
        :math: `\tau = \{0.0, \tau_1, \ldots, \tau_k, 1.0\}` where
        :math: `\tau_k` are the quantiles to be estimated by the network.
        The values for :math:`x_0.0` and :math:`x_1.0` are computed using
        .. math::
            x_0.0 = 2.0 x_{\tau_1} - x_{\tau_2}
            x_1.0 = 2.0 x_{\tau_k} - x_{\tau_{k-1}}
        Arguments:
            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.
        Returns:
            Tuple (xs, fs) containing the :math: `x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        y_pred = np.zeros(prediction.shape[0] + 2)
        y_pred[1:-1] = prediction
        y_pred[0] = 2.0 * y_pred[1] - y_pred[2]
        y_pred[-1] = 2.0 * y_pred[-2] - y_pred[-3]

        qs = np.zeros(prediction.shape[0] + 2)
        qs[1:-1] = quantiles
        qs[0] = 0.0
        qs[-1] = 1.0

        return y_pred, qs
    
def posterior_mean(prediction, quantiles):
        import numpy as np
        r"""
        Computes the posterior mean by computing the first moment of the
        estimated posterior CDF.
        Arguments:
            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the posterior mean.
        Returns:
            Array containing the posterior means for the provided inputs.
        """
        y_pred, qs = cdf(prediction, quantiles)
        mus = y_pred[-1] - np.trapz(qs, x=y_pred)
        return mus
def calculate_MSE(mean, yTest):
    import numpy as np
    error = np.zeros((len(yTest),1))
    for i in range(len(yTest)):
        
        error[i,0] = (yTest[i]-mean[i])*(yTest[i]-mean[i])
        
    return error
def calculate_MAE(mean, yTest):
    import numpy as np
    error = np.zeros((len(yTest),1))
    for i in range(len(yTest)):
        
        error[i,0] = np.abs((yTest[i]-mean[i]))
        
    return error
import keras.backend as K
def skewed_absolute_error(y_true, y_pred, tau):
    """
    The quantile loss function for a given quantile tau:
    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)
    Where I is the indicator function.
    """
    #print(y_true)
    dy = y_pred - y_true
    return (1.0 - tau) * K.relu(dy) + tau * K.relu(-dy)


def quantile_loss(y_true, y_pred, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    #print(y_true.shape)
    #print(y_true)
    #print(y_pred)
    #e=0
    #print(K.int_shape(y_pred))
    #y_true = K.flatten(y_true)
    
    #y_pred = K.reshape(y_pred,(*K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]))
    e = skewed_absolute_error(K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
            e += skewed_absolute_error(K.flatten(y_true),
                                       K.flatten(y_pred[:, i + 1]),
                                       tau)
    '''
    for i in range(7):
        for j in range(7):
            
            tmp_y_pred = y_pred[:,i,j,:]
            tmp_y_true = y_true[:,i,j]
            
            e += skewed_absolute_error(
                K.flatten(tmp_y_true), K.flatten(tmp_y_pred[:, 0]), taus[0])
            for i, tau in enumerate(taus[1:]):
                e += skewed_absolute_error(K.flatten(tmp_y_true),
                                           K.flatten(tmp_y_pred[:, i + 1]),
                                           tau)
    '''
    return e

def calculate_crps(prediction,yTest, quantiles):
    import numpy as np
    size = 5
    y_pred = prediction
    y_test = yTest
    y_cdf = np.zeros((y_pred.shape[0], size + 2))
    y_cdf[:, 1:-1] = y_pred
    y_cdf[:, 0] = 2.0 * y_pred[:, 1] - y_pred[:, 2]
    y_cdf[:, -1] = 2.0 * y_pred[:, -2] - y_pred[:, -3]
    
    ind = np.zeros(y_cdf.shape)
    ind[y_cdf > y_test.reshape(-1, 1)] = 1.0

    qs = np.zeros((1, size + 2))
    qs[0, 1:-1] = quantiles
    qs[0, 0] = 0.0
    qs[0, -1] = 1.0
    integrand = (qs - ind)**2.0
    
    for i in range(len(y_cdf)):
        indexes = np.argsort(y_cdf[i,:])
        y_cdf[i,:] = y_cdf[i,indexes]
        integrand[i,:] = integrand[i,:]
        
        
    return  np.trapz(integrand, y_cdf)
def generate_all_results(model,xTest, yTest,yTrain ,quantiles, save, folder_path, case):
    '''
        function for generating evaluations of the models.
        
        input:
            xTest: the input test images
            yTest: the test labels
            yTrain: the train labels
            save: boolean to be true if the model should generate predictions
            and extected value predictions. The predictions are then saved in 
            predictions.npy and mean_values.npy. False if the predictions
            are already present in the folder path.
            case: A string with the title for the case. Is going to apear in the
            title plots.
    '''
    
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    if save:
        # predict
        prediction = model.predict(xTest)
        #prediction = np.square(prediction)
        print(prediction.shape)
        
        # calculate the mean value
        mean = np.zeros((xTest.shape[0],1))
        #mean = np.reshape(prediction[:,2],(len(prediction),1))
        
        for i in range(xTest.shape[0]):
            
            mean[i,0] = model.posterior_mean(xTest[i,:])
       
        
        np.save(folder_path+'predictions.npy', prediction)   
        np.save(folder_path+'mean_values.npy', mean)
        
    else:
        prediction = np.load(folder_path+'predictions.npy')
        mean = np.load(folder_path+'mean_values.npy')
        #mean = np.reshape(prediction[:,2],(len(prediction),1))
        
    #crps = model.crps(prediction, yTest, np.array(quantiles))
    crps = calculate_crps(prediction, yTest, quantiles)
    
   
  
    ## plots
    generateQQPLot(quantiles, yTest, prediction, case)
    scatterplot(mean, yTest,case)
    histogram_plot(mean, yTest, case,folder_path, [0,20])
    histogram_plot(mean, yTest, case,folder_path, [0,0.5])
    #scatterplot_zoom(mean, yTest,case)
    containsTrueValueIntervals(quantiles, yTest, prediction, 1,folder_path,case)
    CISizeDistribution(prediction, case)
    getCIsizeFractionPlot(prediction, yTest,case, 0.0001, 20, folder_path)
    getCIsizeFractionPlot(prediction, yTest,case, 0.5, 20, folder_path)
    plotLabelSizeVsCISize(prediction, yTest, 1, case, folder_path)
    generateCRPSIntervalPlot(crps,quantiles, yTest, prediction, 1)
    pltCRPSPerIntervalSize(crps,quantiles, yTest, prediction, case, 0.5, 20)
    generate_qqplot_for_intervals(quantiles, yTest, prediction, 1)
    plotIntervalPredictions(yTest, mean, 1)
    plot_small_pred_histograms(mean, yTest, case)
    
    print("####################loss###################")
    #rain_indexes = np.where(yTest >0)[0]
    #no_rain_idnexes = np.where(yTest==0)[0]
    
    #pltLossPerIntervalSize(quantiles, yTest, prediction, case, 0.5, 20)
    print("overall %s " % np.mean(tf.Session().run(quantile_loss(np.float32(yTest), prediction, quantiles))))      
    #print("rain %s " % tf.Session().run(quantile_loss(np.float32(yTest[rain_indexes]), prediction[rain_indexes], quantiles)))      
    #print("no_rain %s " % tf.Session().run(quantile_loss(np.float32(yTest[no_rain_idnexes]), prediction[no_rain_idnexes], quantiles)))      
    
    
    print("############ Test set statistics#################")
    print("mean %s" % np.mean(yTest))
    print("median %s" % np.median(yTest))
    print("std %s" % np.std(yTest))
    
    print("############ prediction statistics#################")
    print("mean %s" % np.mean(mean))
    print("median %s" % np.median(mean))
    print("std %s" % np.std(mean))
    
    
    # generate confision matrix, rain no rain
    
    print("#################### confusion matrixes ###############")
          
    for i , quantile in enumerate(quantiles):
        print("Confusion matrx for the %s quantile" % quantile)
        confusionMatrix(yTest,prediction[:,i])
    
    
    # get  the probability of detection
    print("#################### POD ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_POD(prediction[:,i], yTest)))
    print("%s mean: %s" %(quantile, calculate_POD(mean, yTest)))
            
        
    # get  the false alarm ratio
    print("#################### FAR ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_FAR(prediction[:,i], yTest)))
    print("%s mean: %s" %(quantile, calculate_FAR(mean, yTest)))
    
    
    # get the critical sucess index
    print("#################### CSI ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_CSI(prediction[:,i], yTest)))
    print("%s mean: %s" %(quantile, calculate_CSI(mean, yTest)))
    
    
    # get the bias
    print("#################### bias ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_bias(prediction[:,i], yTest)))
    
    print(" bias: %s" %( calculate_bias(mean, yTest)))
    
    #getBiasPerIntervals(mean, yTest,1, folder_path)
    
    #print("the mean bias is %s" % calculate_bias(mean, yTest))
    
    
    # get the total MSE
    
    print("#################### MSE ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_tot_MSE(prediction[:,i], yTest)))
        #print("%s quantile: %s" %(quantile, np.mean(calculate_MSE(prediction[:,i], yTest))))
    print("#################### MAE ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_tot_MAE(prediction[:,i], yTest)))
        #print("%s quantile: %s" %(quantile, np.mean(calculate_MAE(prediction[:,i], yTest))))
    
    
    mse = calculate_MSE(mean, yTest)
    mae = calculate_MAE(mean, yTest)
   
    

    print("mean MSE: %s" % np.mean(mse))
    print("median MSE: %s" %np.median(mse))
    print("std MSE : %s" % np.std(mse))
    print("mean MAE : % s" % np.mean(mae))
    print("median MAE : %s " %np.median(mae))
    print("std MAE : %s" % np.std(mae))
    
    print("the mean total MSE is %s" % calculate_tot_MSE(mean, yTest))
    print("the mean total MAE is %s" % calculate_tot_MAE(mean, yTest))
    # get the error
    
    getMeansSquareError(yTest, mean, 1)
    
    print("################### CI ##################")
    print("CI length variance: %s" % np.var(prediction[:,-1]-prediction[:,0]))
    print("CI length mean: %s" % np.mean(prediction[:,-1]-prediction[:,0]))  
    print("CI length median: %s" % np.median(prediction[:,-1]-prediction[:,0]))        
   
    
    
    # print the kvota of apriori mean to display how much is learnt
    print("################## kvota ####################3")
    print(get_apriori_mean_estimate_kvote(yTest,yTrain, mean))
    print(get_apriori_mean_estimate_kvote_abs_mean(yTest,yTrain, mean))
    print(get_apriori_median_estimate_kvote(yTest,yTrain, mean))
    print(get_apriori_median_estimate_kvote_abs_mean(yTest,yTrain, mean))
    
    print("################## correlations ################")
    #print("labelsize and corect 30 conf interval: %s" % get_correlation_labelsize_prediction(yTest, prediction))
    print("distance between 5 quantile and true vaule and label size: %s" % get_correlation_MSE_labelsize(yTest, prediction))
    print("same as above but for the a priori mean as predictior : %s" % get_correlation_MSEapriori_labelsize(yTest,yTrain, prediction))
    print("correlation target prediction: %s" % correlation_target_prediction(yTest, mean))
    print("################# interval lengths #######")
    print((prediction[:,-1]-prediction[:,0]).sum()/len(prediction))
    
    print("####################### CRPS ########################")
    print("mean crps over the whole intevals: %s" % np.mean(crps))
    print("mean crps over the whole intevals: %s" % np.median(crps))
    
    
   
    
    print("####################### No rain measures ##############")
    no_rain_indexes = np.where(yTest == 0)[0]
    
    print( "MSE : %s" % calculate_tot_MSE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    print( "MAE : %s" % calculate_tot_MAE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    print( "bias : %s" % calculate_bias(mean[no_rain_indexes], yTest[no_rain_indexes]))
    
    print( "mean CRPS : %s" % np.mean(crps[no_rain_indexes]))
    print( "median CRPS : %s" % np.median(crps[no_rain_indexes]))
    
    print("CI length variance: %s" % np.var(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))
    print("CI length mean: %s" % np.mean(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))  
    print("CI length median: %s" % np.median(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))   
    
    print("####################### rain ocurencies measures ##############")
    no_rain_indexes = np.where(yTest > 0)[0]
    
    print( "MSE : %s" % calculate_tot_MSE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    print( "MAE : %s" % calculate_tot_MAE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    
    print( "bias : %s" % calculate_bias(mean[no_rain_indexes], yTest[no_rain_indexes]))
    
    print( "CRPS : %s" % np.mean(crps[no_rain_indexes]))
    print( "median CRPS : %s" % np.median(crps[no_rain_indexes]))
    
    print("CI length variance: %s" % np.var(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))
    print("CI length mean: %s" % np.mean(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))  
    print("CI length median: %s" % np.median(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0])) 
    
def generate_all_results_unet(model,xTest, yTest,yTrain ,quantiles, save, folder_path, case):
    
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    if save:
        # predict
        prediction = model.predict(xTest)
        #prediction = np.square(prediction)
        print(prediction.shape)
        
        # calculate the mean value
        mean = np.zeros((xTest.shape[0],100,100))
        mean = np.reshape(prediction[:,:,:,2],(len(prediction),100,100))
        '''
        for i in range(xTest.shape[0]):
            
            mean[i,:] = model.posterior_mean(xTest[i,:])
       
        '''
        np.save(folder_path+'predictions.npy', prediction)   
        np.save(folder_path+'mean_values.npy', mean)
        
    else:
        prediction = np.load(folder_path+'predictions.npy')
        mean = np.load(folder_path+'mean_values.npy')
        #mean = np.reshape(prediction[:,2],(len(prediction),1))
    
    prediction = np.reshape(prediction, (len(prediction)*100*100,5))
    mean = np.reshape(mean, (len(mean)*100*100,1)) 
    yTest = np.reshape(yTest, (len(yTest)*100*100,1))
    yTrain = np.reshape(yTrain, (len(yTrain)*100*100,1))
    crps = calculate_crps(prediction, yTest, quantiles)
    
   
  
    ## plots
    generateQQPLot(quantiles, yTest, prediction, case)
    scatterplot(mean, yTest,case)
    histogram_plot(mean, yTest, case,folder_path, [0,20])
    histogram_plot(mean, yTest, case,folder_path, [0,0.5])
    #scatterplot_zoom(mean, yTest,case)
    containsTrueValueIntervals(quantiles, yTest, prediction, 1,folder_path,case)
    CISizeDistribution(prediction, case)
    getCIsizeFractionPlot(prediction, yTest,case, 0.0001, 20, folder_path)
    getCIsizeFractionPlot(prediction, yTest,case, 0.5, 20, folder_path)
    plotLabelSizeVsCISize(prediction, yTest, 1, case, folder_path)
    generateCRPSIntervalPlot(crps,quantiles, yTest, prediction, 1)
    pltCRPSPerIntervalSize(crps,quantiles, yTest, prediction, case, 0.5, 20)
    generate_qqplot_for_intervals(quantiles, yTest, prediction, 1)
    plotIntervalPredictions(yTest, mean, 1)
    plot_small_pred_histograms(mean, yTest, case)
    
    print("####################loss###################")
    #rain_indexes = np.where(yTest >0)[0]
    #no_rain_idnexes = np.where(yTest==0)[0]
    
    #pltLossPerIntervalSize(quantiles, yTest, prediction, case, 0.5, 20)
    print("overall %s " % np.mean(tf.compat.v1.Session().run(quantile_loss(np.float32(yTest), prediction, quantiles))))      
    #print("rain %s " % tf.Session().run(quantile_loss(np.float32(yTest[rain_indexes]), prediction[rain_indexes], quantiles)))      
    #print("no_rain %s " % tf.Session().run(quantile_loss(np.float32(yTest[no_rain_idnexes]), prediction[no_rain_idnexes], quantiles)))      
    
    
    print("############ Test set statistics#################")
    print("mean %s" % np.mean(yTest))
    print("median %s" % np.median(yTest))
    print("std %s" % np.std(yTest))
    
    print("############ prediction statistics#################")
    print("mean %s" % np.mean(mean))
    print("median %s" % np.median(mean))
    print("std %s" % np.std(mean))
    
    
    # generate confision matrix, rain no rain
    
    print("#################### confusion matrixes ###############")
          
    for i , quantile in enumerate(quantiles):
        print("Confusion matrx for the %s quantile" % quantile)
        confusionMatrix(yTest,prediction[:,i])
    
    
    # get  the probability of detection
    print("#################### POD ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_POD(prediction[:,i], yTest)))
    print("%s mean: %s" %(quantile, calculate_POD(mean, yTest)))
            
        
    # get  the false alarm ratio
    print("#################### FAR ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_FAR(prediction[:,i], yTest)))
    print("%s mean: %s" %(quantile, calculate_FAR(mean, yTest)))
    
    
    # get the critical sucess index
    print("#################### CSI ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_CSI(prediction[:,i], yTest)))
    print("%s mean: %s" %(quantile, calculate_CSI(mean, yTest)))
    
    
    # get the bias
    print("#################### bias ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_bias(prediction[:,i], yTest)))
    
    print(" bias: %s" %( calculate_bias(mean, yTest)))
    
    #getBiasPerIntervals(mean, yTest,1, folder_path)
    
    #print("the mean bias is %s" % calculate_bias(mean, yTest))
    
    
    # get the total MSE
    
    print("#################### MSE ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_tot_MSE(prediction[:,i], yTest)))
        #print("%s quantile: %s" %(quantile, np.mean(calculate_MSE(prediction[:,i], yTest))))
    print("#################### MAE ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_tot_MAE(prediction[:,i], yTest)))
        #print("%s quantile: %s" %(quantile, np.mean(calculate_MAE(prediction[:,i], yTest))))
    
    
    mse = calculate_MSE(mean, yTest)
    mae = calculate_MAE(mean, yTest)
   
    

    print("mean MSE: %s" % np.mean(mse))
    print("median MSE: %s" %np.median(mse))
    print("std MSE : %s" % np.std(mse))
    print("mean MAE : % s" % np.mean(mae))
    print("median MAE : %s " %np.median(mae))
    print("std MAE : %s" % np.std(mae))
    
    print("the mean total MSE is %s" % calculate_tot_MSE(mean, yTest))
    print("the mean total MAE is %s" % calculate_tot_MAE(mean, yTest))
    # get the error
    
    getMeansSquareError(yTest, mean, 1)
    
    print("################### CI ##################")
    print("CI length variance: %s" % np.var(prediction[:,-1]-prediction[:,0]))
    print("CI length mean: %s" % np.mean(prediction[:,-1]-prediction[:,0]))  
    print("CI length median: %s" % np.median(prediction[:,-1]-prediction[:,0]))        
   
    
    
    # print the kvota of apriori mean to display how much is learnt
    print("################## kvota ####################3")
    print(get_apriori_mean_estimate_kvote(yTest,yTrain, mean))
    print(get_apriori_mean_estimate_kvote_abs_mean(yTest,yTrain, mean))
    print(get_apriori_median_estimate_kvote(yTest,yTrain, mean))
    print(get_apriori_median_estimate_kvote_abs_mean(yTest,yTrain, mean))
    
    print("################## correlations ################")
    #print("labelsize and corect 30 conf interval: %s" % get_correlation_labelsize_prediction(yTest, prediction))
    print("distance between 5 quantile and true vaule and label size: %s" % get_correlation_MSE_labelsize(yTest, prediction))
    print("same as above but for the a priori mean as predictior : %s" % get_correlation_MSEapriori_labelsize(yTest,yTrain, prediction))
    print("correlation target prediction: %s" % correlation_target_prediction(yTest, mean))
    print("################# interval lengths #######")
    print((prediction[:,-1]-prediction[:,0]).sum()/len(prediction))
    
    print("####################### CRPS ########################")
    print("mean crps over the whole intevals: %s" % np.mean(crps))
    print("mean crps over the whole intevals: %s" % np.median(crps))
    
    
   
    
    print("####################### No rain measures ##############")
    no_rain_indexes = np.where(yTest == 0)[0]
    
    print( "MSE : %s" % calculate_tot_MSE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    print( "MAE : %s" % calculate_tot_MAE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    print( "bias : %s" % calculate_bias(mean[no_rain_indexes], yTest[no_rain_indexes]))
    
    print( "mean CRPS : %s" % np.mean(crps[no_rain_indexes]))
    print( "median CRPS : %s" % np.median(crps[no_rain_indexes]))
    
    print("CI length variance: %s" % np.var(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))
    print("CI length mean: %s" % np.mean(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))  
    print("CI length median: %s" % np.median(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))   
    
    print("####################### rain ocurencies measures ##############")
    no_rain_indexes = np.where(yTest > 0)[0]
    
    print( "MSE : %s" % calculate_tot_MSE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    print( "MAE : %s" % calculate_tot_MAE(mean[no_rain_indexes], yTest[no_rain_indexes]))
    
    print( "bias : %s" % calculate_bias(mean[no_rain_indexes], yTest[no_rain_indexes]))
    
    print( "CRPS : %s" % np.mean(crps[no_rain_indexes]))
    print( "median CRPS : %s" % np.median(crps[no_rain_indexes]))
    
    print("CI length variance: %s" % np.var(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))
    print("CI length mean: %s" % np.mean(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0]))  
    print("CI length median: %s" % np.median(prediction[no_rain_indexes,-1]-prediction[no_rain_indexes,0])) 
  
def generate_all_results_CNN(prediction,mean,xTest, yTest,yTrain, quantiles):
    
    import numpy as np
    
    # predict
    #prediction = model.predict(xTest)
    
    #calculate the mean value
    '''
    mean = np.zeros((xTest.shape[0],1))
    for i in range(xTest.shape[0]):
        
        mean[i,0] = posterior_mean(prediction,quantiles)
    
    
    '''
    # generate QQ plot
    generateQQPLot(quantiles, yTest, prediction)
    
    # generate qq plots for intervals of y data
    generate_qqplot_for_intervals(quantiles, yTest, prediction, 1)
    
    # get the error
    getMeansSquareError(yTest, mean, 1)
    
    # generate confision matrix, rain no rain
    print("#################### confusion matrixes ###############")
          
    for i , quantile in enumerate(quantiles):
        print("Confusion matrx for the %s quantile" % quantile)
        confusionMatrix(yTest,prediction[:,i])
    
    # plot interval predictions
    plotIntervalPredictions(yTest, mean, 1)
    
    # get  the probability of detection
    print("#################### POD ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_POD(prediction[:,i], yTest)))
        
    # get  the false alarm ratio
    print("#################### FAR ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_FAR(prediction[:,i], yTest)))
    
    
    # get the critical sucess index
    print("#################### CSI ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_CSI(prediction[:,i], yTest)))
    
    
    # get the bias
    print("#################### bias ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_bias(prediction[:,i], yTest)))
    
    print("the mean bias is %s" % calculate_bias(mean, yTest))
    
    # get the total MSE
    print("#################### MSE ###############")
    for i, quantile in enumerate(quantiles):
        print("%s quantile: %s" %(quantile, calculate_tot_MSE(prediction[:,i], yTest)))
    print("the mean total MSE is %s" % calculate_tot_MSE(mean, yTest))
    
    # print the kvota of apriori mean to display how much is learnt
    print("################## kvota ####################3")
    print(get_apriori_mean_estimate_kvote(yTest,yTrain, mean))
    print(get_apriori_mean_estimate_kvote_abs_mean(yTest,yTrain, mean))
    
    
    print("################## correlations ################")
    #print("labelsize and corect 30 conf interval: %s" % get_correlation_labelsize_prediction(yTest, prediction))
    #print("distance between 5 quantile and true vaule and label size: %s" % get_correlation_MSE_labelsize(yTest, prediction))
    #print("same as above but for the a priori mean as predictior : %s" % get_correlation_MSEapriori_labelsize(yTest,yTrain, prediction))
    print("correlation target prediction: %s" % correlation_target_prediction(yTest, prediction))
    
    print("################# interval lengths #######")
    print((prediction[:,-1]-prediction[:,0]).sum()/len(prediction))
    print("################# interval lengths #######")
    print((prediction[:,-1]-prediction[:,0]).sum()/len(prediction))
    
    
def tmp_code_mean_vspixel():
    import numpy as np
    xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
    yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
    times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
    distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
    #xData = xData[:,:,4:25,4:25]
    
    '''
    xData =np.load(folder_path+'/trainingData/xDataC8C13S700000_R28_P400GPM_res3interval_3.npy')
    yData = np.load(folder_path+'/trainingData/yDataC8C13S700000_R28_P400GPM_res3interval_3.npy')
    times = np.load(folder_path+'/trainingData/timesC8C13S700000_R28_P400GPM_res3interval_3.npy')
    distance = np.load(folder_path+'/trainingData/distanceC8C13S700000_R28_P400GPM_res3interval_3.npy') 
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
    
    
    print(xData.shape)
    
    from typhon.retrieval import qrnn
    # = qrnn.QRNN()
    model = qrnn.QRNN.load('model.h5')
    
    
    
    
    xData2 =np.load('trainingData/xDataC8C13S3200_R28_P4GPM_res3.npy')
    yData2 = np.load('trainingData/yDataC8C13S3200_R28_P4GPM_res3.npy')
    times2 = np.load('trainingData/timesC8C13S3200_R28_P4GPM_res3.npy')
    distance2 = np.load('trainingData/distanceC8C13S3200_R28_P4GPM_res3.npy')  
    
    # remove nan values
    
    nanValues =np.argwhere(np.isnan(xData2)) 
    xData2 = np.delete(xData2,np.unique(nanValues[:,0]),0)
    yData2 = np.delete(yData2,np.unique(nanValues[:,0]),0)
    times2 = np.delete(times2,np.unique(nanValues[:,0]),0)
    distance2 = np.delete(distance2,np.unique(nanValues[:,0]),0)
    
    
    print(xData2.shape)
    print(yData2.shape)
    print(times2.shape)
    print(distance2.shape)
    
    x_data_test = np.zeros((3200,28,28,2))
    x_data_test[:,:,:,0] = (xData2[:,0,3,3,:,:]-min1)/(max1-min1)
    x_data_test[:,:,:,1] = (xData2[:,1,3,3,:,:]-min2)/(max2-min2)
    times_test = times2[:,:,3,3]
    distance_test = distance2[:,:,3,3]
    
    
    newXData = np.reshape(x_data_test,(x_data_test.shape[0],x_data_test.shape[1]*x_data_test.shape[2]*x_data_test.shape[3]))
    tmp = np.zeros((x_data_test.shape[0],x_data_test.shape[1]*x_data_test.shape[2]*x_data_test.shape[3]+4))
    tmp[:,:x_data_test.shape[1]*x_data_test.shape[2]*x_data_test.shape[3]] = newXData
    
    
    tmp[:,-1] = (times_test[:,0]-times_test[:,1])/1000
    tmp[:,-2] = distance_test[:,0]
    tmp[:,-3] = distance_test[:,1]
    tmp[:,-4] = (times_test[:,0]-times_test[:,2])/1000
    
    predictions = model.predict(tmp)
    
    
    tmp_y_train = np.mean(yData[:175000,:,:], axis=(1,2))
    tmp_y_test = np.mean(yData2,axis=(1,2))
    
    print(tmp_y_train.shape)
    print(tmp_y_test.shape)
    
    from visulize_results import generate_all_results_CNN
    quantiles = [0.1,0.3,0.5,0.7,0.9]
    generate_all_results_CNN(predictions,np.reshape(predictions[:,2],(len(predictions[:,2]),1)),None,tmp_y_test,tmp_y_train, quantiles)

    