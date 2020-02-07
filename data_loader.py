# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:52:46 2020

@author: gustav
"""

# Constants
rad = []
receptiveField = 28
maxLongitude = -51
minLongitde = -70
maxLatitide = 2.5
minLatitude = -11
lons = []
lats = []
oldFile = ""

def getFilesForHour(DATE):
    
    '''
        Returns an iterator of file names for the specific hour defines by DATE
    '''
    from google.cloud import storage
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    PATH_Storage = 'ABI-L1b-RadF/%s' % (DATE.strftime('%Y/%j/%H')) 
    
    return bucket.list_blobs(prefix = PATH_Storage)
def getMiddleTime(FILE):
    import datetime
    middleTimeDifference =(datetime.datetime.strptime(FILE.split('_')[4], 'e%Y%j%H%M%S%f')-datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')).total_seconds()
    return (datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')+datetime.timedelta(0, int(middleTimeDifference/2)))

def getClosestFile(DATE, CHANNEL):
    from datetime import datetime
    import numpy as np
    '''
    Returns the filepath closest to the DATE object
    
    '''
    files_for_hour = list(map(lambda x : x.name, getFilesForHour(DATE)))
   
    files_for_hour = [file for file in files_for_hour if file.split('_')[1][-3:] == CHANNEL ]
    if len(files_for_hour) == 0:
        print("could npt get closest file"+str(DATE))
     
    
    date_diff = np.zeros((len(files_for_hour),1))
    
    for i in range(len(files_for_hour)):
        
        middleTime = getMiddleTime(files_for_hour[i])
        
        date_diff[i] = np.abs((DATE-middleTime).total_seconds())
   
    return '%s' % (files_for_hour[np.argmin(date_diff[:,0])]), getMiddleTime(files_for_hour[np.argmin(date_diff[:,0])]) 

def downloadFile(FILE):
    '''
    Downloads FILE and saves it in the data folder
    '''
    from google.cloud import storage
    #check if file lready exists
    
    try:
        f = open('data/'+FILE.split('/')[-1])
        # Do something with the file
        return
    except IOError:
        print("File does not exist, downloading")
   
        
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    blob = bucket.blob(FILE)
    blob.download_to_filename('data/'+FILE.split('/')[-1])
    
def extractGeoData(filePATH):
    import xarray
    from pyproj import Proj
    import numpy as np
    '''
    global lons,lats
    global rad
    global x_data
    global y_data
    global sat_h
    global sat_lon
    global sat_sweep
    '''
    maxLongitude = -40
    minLongitde = -80
    maxLatitide = 8
    minLatitude = -20
    
    
    FILE = 'data/'+filePATH.split('/')[-1]
    
    C = xarray.open_dataset(FILE)
    
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
    
   
    xmin_proj =min([xmin_proj,xmax_proj])
    xmax_proj =max([xmin_proj,xmax_proj])
    ymin_proj =min([ymin_proj,ymax_proj])
    ymax_proj =max([ymin_proj,ymax_proj])
    
    
    
    
    xmin_index = (np.abs(x_new.data-xmin_proj)).argmin()
    xmax_index = (np.abs(x_new.data-xmax_proj)).argmin()
    ymin_index = (np.abs(y_new.data-ymin_proj)).argmin()
    ymax_index = (np.abs(y_new.data-ymax_proj)).argmin()
     
    
    x_new = x_new[xmin_index:xmax_index]
    y_new = y_new[ymax_index:ymin_index]
   
    x_new.coords['x'] = x_new.coords['x']* sat_h
    y_new.coords['y'] = y_new.coords['y']* sat_h
    
    rad = rad[ymax_index:ymin_index,xmin_index:xmax_index]
    rad.coords['x'] =rad.coords['x']*sat_h
    rad.coords['y'] =rad.coords['y']*sat_h
    
    x = x_new
    y = y_new
    
    x_data = x
    y_data = y
    # Perform cartographic transformation. That is, convert image projection coordinates (x and y)
    # to latitude and longitude values.
    XX, YY = np.meshgrid(x, y)
    print("traonsfrming data")
    lons, lats = p(XX, YY, inverse=True)
    #print(lons)
    print("transformation done")
    return lons,lats,C,rad, x_data, y_data

def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude, sat_h, sat_lon, sat_sweep, x_data,y_data):
    # project the longitude and latitude to geostationary references
    from pyproj import Proj
    import numpy as np
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    x, y = p(longitude, latitude)
    
    
    # get the x and y indexes
    x_index = (np.abs(x_data.data-x)).argmin()
    y_index = (np.abs(y_data.data-y)).argmin()
    
    lons, lats = p(x_data[x_index], y_data[y_index], inverse=True)
  
    return y_index, x_index, np.sqrt((longitude-lons)*(longitude-lons)+(latitude-lats)*(latitude-lats))
    
def getGEOData(GPM_data, dataSize):
    from netCDF4 import Dataset
    import numpy as np
    import time
    from datetime import datetime

    
    #if longitude < minLongitde or longitude > maxLongitude or latitude < minLatitude or latitude >maxLatitide:
    #    return np.zeros((receptiveField,receptiveField))
    
    # Download the data file
    filePaths = []
    newFileIndexes = []
    previousFileName = ""
    start_time = time.time()
 
   
    middleTime = datetime(1980,1,1)
    
    for i in range(len(GPM_data[:dataSize,3])):
        currentTime = convertTimeStampToDatetime(GPM_data[i,3])
        if(np.abs((currentTime-middleTime).total_seconds()) > 600):
            
            filePath, middleTime = getClosestFile(currentTime, 'C13')
           
        if filePath != previousFileName:
            newFileIndexes.append(i)
            filePaths.append(filePath)
            previousFileName = filePath
        
    end_time = time.time()
    print("time for getting file paths %s" % (end_time-start_time))
    # iterate through all unique file names
    xData = np.zeros((dataSize,receptiveField,receptiveField))
    times = np.zeros((dataSize,1))
    distance = np.zeros((dataSize,1))
    
    for i in range(len(newFileIndexes)):
        
        filePATH = filePaths[i]
        FILE = 'data/'+filePaths[i].split('/')[-1]
        print(FILE)
       
        downloadFile(filePATH)
        
        
       
        lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH)
      
       
        
        sat_h = C['goes_imager_projection'].perspective_point_height
        
        # Satellite longitude
        sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
        
        # Satellite sweep
        sat_sweep = C['goes_imager_projection'].sweep_angle_axis
       
        if i == len(newFileIndexes)-1:
            endIndex = dataSize
        else:
            endIndex = newFileIndexes[i+1]
        
        
        
        for j in range(newFileIndexes[i],endIndex):
            xIndex, yIndex , distance[j,0]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,1], GPM_data[j,2], sat_h, sat_lon, sat_sweep, x_data,y_data)
            xData[j,:,:], times[j,0] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
            
        
    
    return xData, np.reshape(times,(len(times))), distance   

def getGPMFilesForSpecificDay(DATE):
    '''
        returning a list of file name for that spcific date
    '''
    import http.client
    import re
    c = http.client.HTTPSConnection("gpm1.gesdisc.eosdis.nasa.gov")

    request_string = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2BCMB.06/%s/' % (DATE.strftime('%Y/%j'))
    c.request("GET", request_string)
    r = c.getresponse()
    r = str(r.read())
   
    files = list(set(re.compile('"[^"]*.HDF5"').findall(r)))
  
    files.sort()
    return [f[1:-1] for f in files]

def downloadGPMFile(FILENAME, DATE):
    
    '''
        downloading rain totals form filename
    '''
    # check if file lready exists
    try:
        f = open('data/'+FILENAME)
        # Do something with the file
        return
    except IOError:
        print("File does not exist, downloading")
        
    maxLongitude = -51
    minLongitude = -70
    maxLatitude = 2.5
    minLatitude = -11
    host_name = 'https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?'
    filename = '%2Fdata%2FGPM_L2%2FGPM_2BCMB.06%2F' + DATE.strftime('%Y') + '%2F' + DATE.strftime('%j') + '%2F' + FILENAME
    p_format =  'aDUv'
    bbox = str(minLatitude) + '%2C' + str(minLongitude) + '%2C' + str(maxLatitude) + '%2C' + str(maxLongitude)
    label = FILENAME[:-8]+'SUB.HDF5'
    flags = 'GRIDTYPE__SWATH'
    variables = '..2FNS..2FsurfPrecipTotRate%2C..2FNS..2Fnavigation..2FtimeMidScanOffset'
    #URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FGPM_L2%2FGPM_2BCMB.06%2F2019%2F007%2F2B.GPM.DPRGMI.CORRA2018.20190107-S194520-E211752.027616.V06A.HDF5&FORMAT=aDUv&BBOX=-70%2C-180%2C70%2C180&LABEL=2B.GPM.DPRGMI.CORRA2018.20190107-S194520-E211752.027616.V06A.SUB.HDF5&FLAGS=GRIDTYPE__SWATH&SHORTNAME=GPM_2BCMB&SERVICE=SUBSET_LEVEL2&VERSION=1.02&DATASET_VERSION=06&VARIABLES=..2FNS..2FsurfPrecipTotRate%2C..2FNS..2Fnavigation..2FtimeMidScanOffset'
    URL = host_name + 'FILENAME=' + filename + '&FORMAT=' + p_format + '&BBOX=' + bbox + '&LABEL=' + label + '&FLAGS=' + flags + '&SHORTNAME=GPM_2BCMB&SERVICE=SUBSET_LEVEL2&VERSION=1.02&DATASET_VERSION=06&VARIABLES=' + variables
    
    
    import requests
    result = requests.get(URL)
    try:
       result.raise_for_status()
       f = open('data/' + FILENAME,'wb')
       f.write(result.content)
       f.close()
       print('contents of URL written to '+FILENAME)
    except:
       print('requests.get() returned an error code '+str(result.status_code))
       
def getGPMData(start_DATE, maxDataSize, data_per_GPM_pass, rain_norain_division):
    
    '''
        retruns GPM data for the day provided. The data is in form of an array
        with each entry having attributes:
            1: position
            2: time
            3: rain amount
    '''
    
    import h5py
    import numpy as np
    from datetime import datetime, timedelta
    import random
    days_missing_in_GEO = [str(datetime(2017,8,3)),
                           str(datetime(2017,9,26)),
                           str(datetime(2017,9,27)),
                           str(datetime(2017,9,28)),
                           str(datetime(2017,9,29)),
                           str(datetime(2017,11,30)),
                           str(datetime(2017,12,1)),
                           str(datetime(2017,12,2)),
                           str(datetime(2017,12,3)),
                           str(datetime(2017,12,4)),
                           str(datetime(2017,12,5)),
                           str(datetime(2017,12,6)),
                           str(datetime(2017,12,7)),
                           str(datetime(2017,12,8)),
                           str(datetime(2017,12,9)),
                           str(datetime(2017,12,10)),
                           str(datetime(2017,12,11)),
                           str(datetime(2017,12,12)),
                           str(datetime(2017,12,13)),
                           str(datetime(2017,12,14)),
                           str(datetime(2018,1,28)),
                           str(datetime(2018,2,21)),
                           str(datetime(2018,2,22)),
                           ]
    data = np.zeros((maxDataSize,4))
    
    # get the files for the specific day
    
    DATE = start_DATE
    index = 0
    while index < maxDataSize:
        
        
        if str(DATE) in days_missing_in_GEO:
            DATE += timedelta(days=1)
            continue
        
        files = getGPMFilesForSpecificDay(DATE)
        
        for FILENAME in files:
            # download the gpm data
            downloadGPMFile(FILENAME,DATE)
            
            # read the data
            try:
                path =  'data/'+FILENAME
                f = h5py.File(path, 'r')
                precepTotRate = f['NS']['surfPrecipTotRate'][:].flatten()
                longitude = f['NS']['Longitude'][:].flatten()
                latitude = f['NS']['Latitude'][:].flatten()
                time = np.array([f['NS']['navigation']['timeMidScan'][:],]*f['nrayNS_idx'].shape[0]).transpose().flatten()
            except:
                continue
            # remove null data 
            indexes = np.where(np.abs(precepTotRate) < 200)[0]
            precepTotRate = precepTotRate[indexes]
            longitude = longitude[indexes]
            latitude = latitude[indexes]
            time = time[indexes]
            
            # get indexes of rainy data
            
            rain_indexes = np.where(np.abs(precepTotRate) > 0)[0]
            norain_indexes = np.where(np.abs(precepTotRate) == 0)[0]
            '''
            print(len(precepTotRate))
            print(len(rain_indexes))
            print(len(norain_indexes))
            print(int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division))
            print(len( norain_indexes[:int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division)]))
            '''
            tot_indexes = np.concatenate((rain_indexes, norain_indexes[:int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division)]))
            #tot_indexes = rain_indexes + norain_indexes[:int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division)]
            #print(len(tot_indexes))
            precepTotRate = precepTotRate[tot_indexes]
            longitude = longitude[tot_indexes]
            latitude = latitude[tot_indexes]
            time = time[tot_indexes]
            
            
            
            index1 = min(maxDataSize,index+len(precepTotRate),index+data_per_GPM_pass)
            # select random data
            
            indexes = random.sample(range(0, len(precepTotRate)), index1-index)
            
           
            
            data[index:index1,0] = precepTotRate[indexes]
            data[index:index1,1] = longitude[indexes]
            data[index:index1,2] = latitude[indexes]
            data[index:index1,3] = time[indexes]
            
            index = index1
            print(index)
            if index > maxDataSize:
                break
            
           
                
        
        DATE += timedelta(days=1)
        print(DATE)
      
    return data
   
def convertTimeStampToDatetime(timestamp):
    from datetime import datetime, timedelta
    return datetime(1980, 1, 6) + timedelta(seconds=timestamp - (35 - 19))
    

def getTrainingData(dataSize, nmb_GPM_pass, rain_norain_division):
    
    import numpy as np
    import datetime
    import time
    #receptiveField = 28

    '''
    returns a set that conisit of radiance data for an area around every pixel
    in the given area together with its label
    '''
    
    xData = np.zeros((dataSize,receptiveField,receptiveField))
    times = np.zeros((dataSize,2))
    yData = np.zeros((dataSize,1))
    '''
        First step is to get the label data. To do this, we look at a specifi
        passing of the satelite over the area. We then extract the points
        in wich it passes. the result is a list of all the pixel that it 
        passet. Each entry in the list has the following atributes
            
            1: position
            2: time
            3: rain amount
    '''
    start_time = time.time()
    GPM_data = getGPMData(datetime.datetime(2017,8,2),dataSize,nmb_GPM_pass,rain_norain_division)
    times[:,0] = GPM_data[:,3]
    end_time = time.time()
    print("time for collecting GPM Data %s" % (end_time-start_time))
    '''
    next step is to pair the label with the geostattionary data.
    '''
    start_time = time.time()
    
    xData[:dataSize,:,:], times[:dataSize,1], distance = getGEOData(GPM_data, dataSize)
    yData = GPM_data[:dataSize,0]
    end_time = time.time()
    
    print("time for collecting GEO Data %s" % (end_time-start_time))
    import numpy as np
    np.save('trainingData/xDataS'+str(dataSize)+'_R28_P'+str(nmb_GPM_pass)+'_R'+str(rain_norain_division)+'.npy', xData)   
    np.save('trainingData/yDataS'+str(dataSize)+'_R28_P'+str(nmb_GPM_pass)+'_R'+str(rain_norain_division)+'.npy', yData)   
    np.save('trainingData/timesS'+str(dataSize)+'_R28_P'+str(nmb_GPM_pass)+'_R'+str(rain_norain_division)+'.npy', times)
    np.save('trainingData/distanceS'+str(dataSize)+'_R28_P'+str(nmb_GPM_pass)+'_R'+str(rain_norain_division)+'.npy', distance)  
    return xData, yData, times, distance


def plotTrainingData(xData,yData, times, nmbImages):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(5800,5800+nmbImages):
        
        fig = plt.figure()
        fig.suptitle('timediff %s, rainfall %s' % (np.abs(times[i,0]-times[i,1]), yData[i]), fontsize=20)
        plt.imshow(xData[i,:,:])
        

def preprocessDataForTraining(xData, yData, times, distance):
    scaler1 = StandardScaler()

    # reshape data for the QRNN
    newXData = np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]))
    newYData = np.reshape(yData,(len(yData),1))
    
    # comine the IR images and the distance and time difference
    tmp = np.zeros((xData.shape[0],xData.shape[1]*xData.shape[2]+2))
    tmp[:,:xData.shape[1]*xData.shape[2]] = newXData
    tmp[:,-1] = times[:,0]-times[:,1]
    tmp[:,-2] = distance[:,0]
    newXData = tmp
    
    # scale the data with unit variance and and between 0 and 1 for the labels
    scaler1.fit(newXData)
    newXData = scaler1.transform(newXData)
    newYData = newYData/newYData.max()
    
    return newXData, newYData