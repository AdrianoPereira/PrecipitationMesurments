# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:14:40 2020

@author: gustav
This file contains all the function required for creating datasets of GOES infraded images paired with GPM rain rate data.

"""
def get_middle_time(FILE):
    '''
        returns the middle time of the GEOES full time scan
    '''
    import datetime
    
    middleTimeDifference =(datetime.datetime.strptime(FILE.split('_')[4], 'e%Y%j%H%M%S%f')-datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')).total_seconds()
    return (datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')+datetime.timedelta(0, int(middleTimeDifference/2)))

def convert_time_stamp_to_datetime(timestamp, time_reference = 'old'):
    
    from datetime import datetime, timedelta
    if time_reference == 'POSIX':
        return datetime.fromtimestamp(timestamp)
    else:
        return datetime(1980, 1, 6) + timedelta(seconds=timestamp - (35 - 19))
    
def get_timestamp(file):
    return get_middle_time(file)

def get_files_for_hour(DATE):
    
    '''
        Returns an iterator of file names for GOES for the specific hour defines by DATE
    '''
    from google.cloud import storage
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    PATH_Storage = 'ABI-L1b-RadF/%s' % (DATE.strftime('%Y/%j/%H')) 
    
    return bucket.list_blobs(prefix = PATH_Storage)

def create_list_of_files(CHANNEL, start_date, end_date):
    from datetime import datetime, timedelta
    from linkedlist import SLinkedList, Node
    import numpy as np
    import pickle
    
    '''
    function for creating a sorted list with all the file name sorted by time. saves the results in a text file
    '''
    
    current_date = start_date
    
    files = []
    i =0
    
    
    while current_date < end_date:
        files_for_hour = list(map(lambda x : x.name, get_files_for_hour(current_date)))
        files_for_hour = [file for file in files_for_hour if file.split('_')[1][-3:] == CHANNEL ]
        
        if len(files_for_hour) == 0:
            current_date += timedelta(hours=1)
            #print('bull')
            continue
        
        files =files + files_for_hour
            
        current_date += timedelta(hours=1)
        
        if i % 100 == 0:
            print(current_date)
        i+=1
    

    files.sort(key=get_timestamp)
    with open(str(CHANNEL)+start_date.strftime("%m%d%Y%H%M%S")+end_date.strftime("%m%d%Y%H%M%S")+'_files.txt', "wb") as fp:   #Pickling
        pickle.dump(files, fp)    
    
    return files
    

def read_linked_list(CHANNEL,start_date,end_date):
    '''
        reads the file name list of the GOES files. If it doesen't exist, it 
        creates it.
    '''
    
    # datetime(2017,8,1),datetime(2020,4,1)
    import pickle
    from datetime import datetime
    
    try:
        with open(str(CHANNEL)+start_date.strftime("%m%d%Y%H%M%S")+end_date.strftime("%m%d%Y%H%M%S")+'_files.txt', "rb") as fp:   # Unpickling
            return pickle.load(fp)
    except: 
       return create_list_of_files(CHANNEL,start_date, end_date)

def download_GOES_file(FILE, folder_path):
    '''
    Downloads FILE and saves it in the data folder
    '''
    from google.cloud import storage
    #check if file lready exists
    #folder_path = '../../../../../E:/Precipitation_mesurments/data'
    try:
        f = open(folder_path+'/data/'+FILE.split('/')[-1])
        # Do something with the file
        return
    except IOError:
        print("File does not exist, downloading")
   
        
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    blob = bucket.blob(FILE)
    blob.download_to_filename(folder_path+'/data/'+FILE.split('/')[-1])
    
def extract_geo_data(filePATH,folder_path, prev_sat_h = 0, prev_sat_lon = 0, prev_sat_sweep = 0, prev_x = [], prev_y = [], prev_lons =[], prev_lats = []):
    
    '''
        function for extracting the GOES data from the file specifiec by filePATH
    '''
    import xarray
    from pyproj import Proj
    import numpy as np
    import time
    
    maxLongitude = -40
    minLongitde = -80
    maxLatitide = 8
    minLatitude = -20
    
    #start_time = time.time()
   
    
    
    FILE = folder_path+'/data/'+filePATH.split('/')[-1]
    
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
    #end_time = time.time()
    
    #print("time for first section %s" % (end_time-start_time))
    
    #start_time = time.time()
    if prev_sat_h == sat_h and prev_sat_lon == sat_lon and prev_sat_sweep == sat_sweep and (prev_x.data == x.data).all() and (prev_y.data == y.data).all():
        lons = prev_lons
        lats = prev_lats
    else:
      
        XX, YY = np.meshgrid(x, y)
        print("traonsfrming data")
        lons, lats = p(XX, YY, inverse=True)
        print("transformation done")
    #end_time = time.time()
    
    #print("time for checking %s" % (end_time-start_time))
    
    return lons,lats,C,rad, x_data, y_data

def get_index_of_geo_data_matric_from_longitude_latitude(longitude, latitude,proj, x_data,y_data):
    # project the longitude and latitude to geostationary references
    from pyproj import Proj
    import numpy as np
    import time
    
   
    x, y = proj(longitude, latitude)
    
    x_index = (np.abs(x_data.data-x)).argmin()
    y_index = (np.abs(y_data.data-y)).argmin()
    
    
    lons, lats = proj(x_data[x_index], y_data[y_index], inverse=True)
  
    return y_index, x_index, np.sqrt((longitude-lons)*(longitude-lons)+(latitude-lats)*(latitude-lats))
   
def get_GEO_data(GPM_data, channel, GOES_image_width, folder_path,time_reference = 'old'):
    
    '''
        inputs: 
            GPM_data: the gpm data that the GOES images center around. 
                Array of size [data_size,4]. The dimensions represent the 
                following: 0:longitude
                            1: latitude
                            2: timestamp
                            3: rain rate
            channel: the channel of the ABI instrument of the GOES satelite
            time_reference: which time reference is used
            folder_path: folder where GOES data should be downloaded to
            
        outputs:
            xData : the GOES images with size [data_size, GOES_image_width,GOES_image_width,2]
            times: the time stamp of the GOES data
            distance: distance from the center of GOES to the Center of GPM.
    '''
    
    from netCDF4 import Dataset
    import numpy as np
    import time
    from datetime import datetime
    from pyproj import Proj
    
  
    # initilizations of arrays
    filePaths = []
    newFileIndexes = []
    previousFileName = ""
    start_time = time.time()
    
    # The names of the availiable files for the GOES data is saved in a txt file.
    # This is done to speed up the code.
    
    # the followng line reads that file into an array
    start_date = convert_time_stamp_to_datetime(min(GPM_data[:,2])-60*60*48, time_reference)
    end_date = convert_time_stamp_to_datetime(max(GPM_data[:,2])+60*60*48, time_reference)
   
    files = read_linked_list(channel, start_date, end_date)
   
    j = 5
    # main loop of all the GPM data points
    for i in range(len(GPM_data[:,2])):
        
        
        currentTime = convert_time_stamp_to_datetime(GPM_data[i,2], time_reference)
        
        current_distance = np.abs((get_middle_time(files[j])-currentTime).total_seconds())
        
        if i % 100000 == 0:
            print(i)
            
        # get the closest file from the sorted list of file names list
        distance_forward = np.abs((get_middle_time(files[j+1])-currentTime).total_seconds())
    
        if current_distance < distance_forward:
            filePath = files[j]
            continue
        j+=1
        
        while True:
            current_distance = np.abs((get_middle_time(files[j])-currentTime).total_seconds())
            distance_forward = np.abs((get_middle_time(files[j+1])-currentTime).total_seconds())
          
            if current_distance < distance_forward:
                filePath = files[j]
                break
            else:
                j+=1
                
           
        if filePath != previousFileName:
            newFileIndexes.append(i)
            filePaths.append(filePath)
            previousFileName = filePath
            
    print(len(filePaths))
        
    end_time = time.time()
    print("time for getting file paths %s" % (end_time-start_time))
    
    # iterate through all unique file names
    nmb_files = 1
    xData = np.zeros((len(GPM_data),nmb_files,GOES_image_width,GOES_image_width))
    times = np.zeros((len(GPM_data),nmb_files))
    distance = np.zeros((len(GPM_data),nmb_files))
    
    lons = []
    lats = []
    sat_h = 0
    sat_lon = 0
    sat_sweep = 0
    x_data = []
    y_data =  []
    
    for i in range(len(newFileIndexes)):
        
        # closest file
        filePATH = filePaths[i]
        #print(filePATH)
        FILE = 'data/'+filePaths[i].split('/')[-1]
        if i % 100 == 0:
            print(i/len(newFileIndexes))
       
        download_GOES_file(filePATH, folder_path)
        #filePATH, prev_sat_h = 0, prev_sat_lon = 0, prev_sat_sweep = 0, prev_x = [], prev_y = [], prev_lons =[], prev_lats = []
        #start_time = time.time()
        print(filePath)
        lons,lats,C,rad, x_data, y_data = extract_geo_data(filePATH,folder_path,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)
        #end_time = time.time()
        #print("time for extractng geo data %s" % (end_time-start_time))
        sat_h = C['goes_imager_projection'].perspective_point_height
        
        # Satellite longitude
        sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
        
        # Satellite sweep
        sat_sweep = C['goes_imager_projection'].sweep_angle_axis
       
        if i == len(newFileIndexes)-1:
            endIndex = len(GPM_data)
        else:
            endIndex = newFileIndexes[i+1]
        
        
        #start_time = time.time()
        proj = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
        
        for j in range(newFileIndexes[i],endIndex):
            #start_time = time.time()
            xIndex, yIndex , distance[j,0]= get_index_of_geo_data_matric_from_longitude_latitude(GPM_data[j,0], GPM_data[j,1], proj, x_data,y_data)
            #end_time = time.time()
            #print(" time for getting index %s" %(end_time -start_time))
            
            #start_time = time.time()
            xData[j,0,:,:], times[j,0] = rad.data[xIndex-int(GOES_image_width/2):xIndex+int(GOES_image_width/2),yIndex-int(GOES_image_width/2):yIndex+int(GOES_image_width/2)], (get_middle_time(FILE)-datetime(1980,1,6)).total_seconds()
            #end_time = time.time()
            #print("time for extracting data: %s" %(end_time-start_time))
        #end_time = time.time()
        #print("time for for lop %s" %(end_time-start_time))
       
    
    return xData[:,0,:,:], times[:,0], distance[:,0]  
 
def download_GPM_file(FILENAME, DATE, window):
    
    '''
        downloading rain totals from filename
        
        input: FILENAME: name of the file
                DATE: date
                window. the longitude latitude window of interest formatted as [min_lon, max_lon, min_lat, max_lat]
        output: saves the file in the following path: /data/+FILENAME
    '''
    
    # check if file lready exists
    try:
        f = open('data/'+FILENAME)
        # Do something with the file
        return
    except IOError:
        print("File does not exist, downloading")
        
    
    maxLongitude = window[1]
    minLongitude = window[0]
    maxLatitude = window[3]
    minLatitude = window[2]
    
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
       
def get_GPM_files_for_specific_day(DATE):
    '''
        returning a list of GPM file names availiable for the specified DATE.
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


def get_GPM_data(start_DATE, max_data_size, data_per_GPM_pass, window):
    
    '''
        inputs: start_Date: date of the first GPM pass
                data_size: dataset size
                data_per_GPM_pass: sample size of each GPM pass
        outputs:
                gpm_data: array of the GPM data. each entry of the array has 
                        the following infomration
                        1: lonitude coordinate
                        2: latitude coordinate
                        3: time stamp
                        4: precipitation
                
    '''
    
    import h5py
    import numpy as np
    from datetime import datetime, timedelta
    import random
    
    # some days are not availiable in GOES dataset. This is a list of them.
    # This could for sure be done in a more efficient way.
    
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
                           str(datetime(2018,5,6)),
                           str(datetime(2019,2,25)),
                           str(datetime(2019,4,15)),
                           str(datetime(2019,5,30)),
                           str(datetime(2019,6,27)),
                           str(datetime(2019,6,28)),
                           str(datetime(2019,11,6))
                           ]
    gpm_data = np.zeros((max_data_size,4))
   
    
    
    DATE = start_DATE
    index = 0
    
    while index < max_data_size:
        
        
        if str(DATE) in days_missing_in_GEO:
            DATE += timedelta(days=1)
            continue
        
        # get the file names for the specific day
        files = get_GPM_files_for_specific_day(DATE)
        
        
        for FILENAME in files:
            # download the gpm data for the file
            download_GPM_file(FILENAME,DATE, window)
            
            # read the data
            try:
                path =  'data/'+FILENAME
                f = h5py.File(path, 'r')
                #dim_1 = f['NS']['surfPrecipTotRate'][:].shape[0]
                #dim_2 = f['NS']['surfPrecipTotRate'][:].shape[1]
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
            
            
            # calculate how many data points should be selected
            index1 = min(max_data_size,index+len(precepTotRate),index+data_per_GPM_pass)
            
            # get the random sample
            indexes = random.sample(range(0, len(precepTotRate)), index1-index)
       
            
            # append the data to the final array
            gpm_data[index:index1,0] = longitude[indexes]
            gpm_data[index:index1,1] = latitude[indexes]
            gpm_data[index:index1,2] = time[indexes]
            gpm_data[index:index1,3] = precepTotRate[indexes]
            
            index = index1
            print(index)
            if index >= max_data_size:
                break
            
           
                
        # increase the date by a day and repeat
        DATE += timedelta(days=1)
        print(DATE)
      
    return gpm_data

def create_dataset(data_size, 
                   samplesize_per_pass,
                   GOES_image_width,
                   START_DATE,
                   window,
                   folder_path):
    
    import numpy as np
    import datetime
    import time
    '''
        inputs: data_size: The total data size
                samplesize_per_pass: sample size per GPM pass
                GOES_image_width: width of the GOES images
                START_DATE: datetime object of the date that the collecting starts
                window: The longitude latitude window of interest formatted as [min_lon, max_lon, min_lat, max_lat]
                folder_path: path to where the data should be stored
        
        outputs: GOES_images: the GOES images from channel 8 and 13 of the ABI instrument. size: [data_size,GOES_image_width,GOES_image_width,2]
                GPM_rain_rate: the 2BCMB images  size: [data_size]
                times: time of the GOES and GPM
                distances: distance difference of the GOES and GPM
                
                It also saves the numpy arrays locally
                
    '''
    
    # Allocate numpy arrays
    xData = np.zeros((data_size,2,GOES_image_width,GOES_image_width))
    times = np.zeros((data_size,3))
    #yData = np.zeros((dataSize,40,40))
    distance = np.zeros((data_size,2))
    
    
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
    
    #GPM_image, GPM_pos_time = getGPMDataImage(datetime.datetime(2017,8,5), dataSize)
    GPM_data = get_GPM_data(START_DATE,data_size,samplesize_per_pass, window)
    times[:,0] = GPM_data[:,2]
    
    end_time = time.time()
    
    print("time for collecting GPM Data %s" % (end_time-start_time))
    
    '''
    next step is to pair the label with the geostattionary data. This is done for the two channels 8 and 13.
    '''
    start_time = time.time()
    
    tmp_x, tmp_time, tmp_distance =  get_GEO_data( GPM_data,'C13',GOES_image_width, folder_path,)
    xData[:,0,:,:] =tmp_x
    times[:,1] =tmp_time
    distance[:,0] =tmp_distance
    
    #xData[:,1,:,:], times[:,2], distance[:,1] = getGEOData(np.reshape(GPM_pos_time_data, (GPM_pos_time_data.shape[0]*GPM_pos_time_data.shape[2]*GPM_pos_time_data.shape[3],GPM_pos_time_data.shape[1])),'C08')
    tmp_x, tmp_time, tmp_distance =  get_GEO_data( GPM_data,'C08',GOES_image_width, folder_path,)
    xData[:,1,:,:] = tmp_x
    times[:,2] = tmp_time
    distance[:,1] = tmp_distance
    
    yData =  GPM_data[:,3]
    end_time = time.time()
    
    print("time for collecting GEO Data %s" % (end_time-start_time))
    import numpy as np
    
    # save the arrays
    np.save(folder_path+'/trainingData/xDataC8C13S'+str(data_size)+'_R'+str(GOES_image_width)+'_P'+str(samplesize_per_pass)+'GPM_res'+'timeSeries.npy', xData)   
    np.save(folder_path+'/trainingData/yDataC8C13S'+str(data_size)+'_R'+str(GOES_image_width)+'_P'+str(samplesize_per_pass)+'GPM_res'+'timeSeries.npy', yData)   
    np.save(folder_path+'/trainingData/timesC8C13S'+str(data_size)+'_R'+str(GOES_image_width)+'_P'+str(samplesize_per_pass)+'GPM_res'+'timeSeries.npy', times)
    np.save(folder_path+'/trainingData/distanceC8C13S'+str(data_size)+'_R'+str(GOES_image_width)+'_P'+str(samplesize_per_pass)+'GPM_res'+'timeSeries.npy', distance)  
   
    
    return xData, yData, times, distance