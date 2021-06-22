# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:52:46 2020

@author: gustav

This file contains functions mostly for creating the datasets of geostationary images collocated with the GPM labels.
the code is poorly commented and instead of trying to clean everything up I thought it would be a better idea to 
let you have the code and then just let me know what parts of the code is of interest and I will be happy to 
clean those sections up. Just contact me at g.tellwe@gmail.com.
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
folder_path = 'E:/Precipitation_mesurments'


def downloadReferenceDataFromFTPServer():
    from ftplib import FTP
    import os, sys, os.path
    
    
    #ddir='C:\\Data\\test\\'
    #os.chdir(ddir)
    ftp = FTP('server-ftpdsa.cptec.inpe.br')
    
    
    ftp.login('rogerio.batista', 'dsa2013')
    directory = '/Daniel/'
    
    
    ftp.cwd(directory)
    
    filenames = ftp.nlst() # get filenames within the directory
    print(filenames)
    
    
    for filename in filenames:
        local_filename = os.path.join('C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\ReferensData\\', filename)
        file = open(local_filename, 'wb')
        ftp.retrbinary('RETR '+ filename, file.write)
    
        file.close()
    
    ftp.quit()
    
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
    '''
        returns the middle time of the GEOES full time scan
    '''
    import datetime
    
    middleTimeDifference =(datetime.datetime.strptime(FILE.split('_')[4], 'e%Y%j%H%M%S%f')-datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')).total_seconds()
    return (datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')+datetime.timedelta(0, int(middleTimeDifference/2)))

def getClosestFile(DATE, CHANNEL):
    from datetime import datetime, timedelta
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
    
    # get the file before and after as well
    index_closest_file = np.argmin(date_diff[:,0])
    
    return '%s' % (files_for_hour[index_closest_file]), getMiddleTime(files_for_hour[np.argmin(date_diff[:,0])]) 

    #return '%s' % (file_before),'%s' % (files_for_hour[index_closest_file]),'%s' % (file_after), getMiddleTime(files_for_hour[np.argmin(date_diff[:,0])]) 

def get_timestamp(file):
    return getMiddleTime(file)

def create_list_of_files(CHANNEL):
    from datetime import datetime, timedelta
    from linkedlist import SLinkedList, Node
    import numpy as np
    import pickle
    
    '''
    function for creating a linked list with all the file name sorted by time. saves the results in a text file
    '''
    
    current_date = datetime(2017,8,1)
    list1 = SLinkedList()
    
    files_for_hour = list(map(lambda x : x.name, getFilesForHour(current_date)))
   
    files_for_hour = [file for file in files_for_hour if file.split('_')[1][-3:] == CHANNEL ]
  
    list1.headval = Node(getMiddleTime(files_for_hour[0]),files_for_hour[0])

    
    files = []
    i =0
    while current_date < datetime(2020,4,1):
        files_for_hour = list(map(lambda x : x.name, getFilesForHour(current_date)))
   
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
        
    with open(CHANNEL+'_files.txt', "wb") as fp:   #Pickling
        pickle.dump(files, fp)     
    


def read_linked_list(CHANNEL):
    import pickle
    with open(CHANNEL+'_files.txt', "rb") as fp:   # Unpickling
        return pickle.load(fp)
       
        
        

def downloadFile(FILE):
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
    
def extractGeoData(filePATH, prev_sat_h = 0, prev_sat_lon = 0, prev_sat_sweep = 0, prev_x = [], prev_y = [], prev_lons =[], prev_lats = []):
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

#def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude, sat_h, sat_lon, sat_sweep, x_data,y_data):
    
def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude,proj, x_data,y_data):
    # project the longitude and latitude to geostationary references
    from pyproj import Proj
    import numpy as np
    import time
    
    #start_time = time.time()
    
    #p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    x, y = proj(longitude, latitude)
    #x = 0
    #y = 0
    #end_time = time.time()
    
    #print("time_for_projection %s" % (end_time-start_time))
    
    #start_time = time.time()
    # get the x and y indexes
    x_index = (np.abs(x_data.data-x)).argmin()
    y_index = (np.abs(y_data.data-y)).argmin()
    #end_time = time.time()
    
    #print("time for finding the index : %s" % (end_time-start_time))
    
    lons, lats = proj(x_data[x_index], y_data[y_index], inverse=True)
  
    return y_index, x_index, np.sqrt((longitude-lons)*(longitude-lons)+(latitude-lats)*(latitude-lats))
    #return 50, 50, 0
    
def getGEODataTimeSeries(GPM_data, channel):
    from netCDF4 import Dataset
    import numpy as np
    import time
    from datetime import datetime
    from pyproj import Proj
    
    filePaths = []
    newFileIndexes = []
    previousFileName = ""
    start_time = time.time()
    files = read_linked_list(channel)
   
    j = 5

    for i in range(len(GPM_data[:,2])):
        
        
        currentTime = convertTimeStampToDatetime(GPM_data[i,2])
        current_distance = np.abs((getMiddleTime(files[j])-currentTime).total_seconds())
        
        if i % 100000 == 0:
            print(i)
        # get the closest file from the linked list
        
        distance_forward = np.abs((getMiddleTime(files[j+1])-currentTime).total_seconds())
        #print('first_forward_distance')
        #print(getMiddleTime(files[j+1]))
        #print(distance_forward)
        if current_distance < distance_forward:
            filePath = [files[j-1],files[j],files[j+1]]
            continue
        j+=1
        
        while True:
            
            current_distance = np.abs((getMiddleTime(files[j])-currentTime).total_seconds())
            distance_forward = np.abs((getMiddleTime(files[j+1])-currentTime).total_seconds())
            
            if current_distance < distance_forward:
                filePath = [files[j-1],files[j],files[j+1]]
                break
            else:
                j+=1
                
           
        if filePath[1] != previousFileName:
            newFileIndexes.append(i)
            filePaths.append(filePath)
            previousFileName = filePath[1]
    print(len(filePaths))
        
    end_time = time.time()
    print("time for getting file paths %s" % (end_time-start_time))
    # iterate through all unique file names
    nmb_files = 3
    xData = np.zeros((len(GPM_data),nmb_files,receptiveField,receptiveField))
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
        for k in range(3):
            
            filePATH = filePaths[i][k]
            FILE = 'data/'+filePaths[i][k].split('/')[-1]
            if i % 100 == 0:
                print(i/len(newFileIndexes))
           
            downloadFile(filePATH)
            #filePATH, prev_sat_h = 0, prev_sat_lon = 0, prev_sat_sweep = 0, prev_x = [], prev_y = [], prev_lons =[], prev_lats = []
            #start_time = time.time()
            lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)
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
                xIndex, yIndex , distance[j,k]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,0], GPM_data[j,1], proj, x_data,y_data)
                #end_time = time.time()
                #print(" time for getting index %s" %(end_time -start_time))
                
                #start_time = time.time()
                xData[j,k,:,:], times[j,k] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
             
    
    return xData[:,:,:,:], times[:,:], distance[:,:]   
def getGEOData(GPM_data, channel, time_reference = 'old'):
    from netCDF4 import Dataset
    import numpy as np
    import time
    from datetime import datetime
    from pyproj import Proj
    
    #if longitude < minLongitde or longitude > maxLongitude or latitude < minLatitude or latitude >maxLatitide:
    #    return np.zeros((receptiveField,receptiveField))
    
    # Download the data file
    #file_paths_before = []
    #file_paths_after = []
    filePaths = []
    newFileIndexes = []
    previousFileName = ""
    start_time = time.time()
    files = read_linked_list(channel)
   
    j = 5
    #found = True
    for i in range(len(GPM_data[:,2])):
        
        
        currentTime = convertTimeStampToDatetime(GPM_data[i,2], time_reference)
        #print(currentTime)
        #print(currentTime)
        current_distance = np.abs((getMiddleTime(files[j])-currentTime).total_seconds())
        #print('Current_time')
        #print(currentTime)
        #print(getMiddleTime(files[j]))
        #print(current_distance)
        #print('Current Time')
        #print(currentTime)
        #if(np.abs((currentTime-middleTime).total_seconds()) > 450):
            
            #file_path_before, filePath, file_path_after, middleTime = getClosestFile(currentTime, channel)
        if i % 100000 == 0:
            print(i)
        # get the closest file from the linked list
        
        distance_forward = np.abs((getMiddleTime(files[j+1])-currentTime).total_seconds())
        #print('first_forward_distance')
        #print(getMiddleTime(files[j+1]))
        #print(distance_forward)
        if current_distance < distance_forward:
            filePath = files[j]
            continue
        j+=1
        
        while True:
            
            current_distance = np.abs((getMiddleTime(files[j])-currentTime).total_seconds())
            distance_forward = np.abs((getMiddleTime(files[j+1])-currentTime).total_seconds())
            #print('while: current time')
            #print(getMiddleTime(files[j]))
            #print(current_distance)
            #print('while: forward time')
            #print(getMiddleTime(files[j+1]))
            #print(distance_forward)
            if current_distance < distance_forward:
                filePath = files[j]
                #print(filePath)
                #print(getMiddleTime(files[j]))
                #print(getMiddleTime(files[j+1]))
                #print(j)
                #print(distance_forward)
                #print(current_distance)
                #found = False
                break
            else:
                j+=1
                
            
            '''   
            new_distance_backward = np.abs((getMiddleTime(files[j-1])-currentTime).total_seconds())
            
            if new_distance < current_distance_forward:
                current_distance = new_distance
                j+=1
            elif   new_distance < current_distance_backward:
                current_distance = new_distance
                j+=1 
            else:
                filePath = files[j]
                #middleTime = node.date 
                break
            '''
        
        #filePath, middleTime = getClosestFile(currentTime, channel)
            #print(file_path_before)
            #print(filePath)
            #print(file_path_after)
            
           
        if filePath != previousFileName:
            newFileIndexes.append(i)
            filePaths.append(filePath)
            #file_paths_before.append(file_path_before)
            #file_paths_after.append(file_path_after)
            previousFileName = filePath
    print(len(filePaths))
        
    end_time = time.time()
    print("time for getting file paths %s" % (end_time-start_time))
    # iterate through all unique file names
    nmb_files = 1
    xData = np.zeros((len(GPM_data),nmb_files,receptiveField,receptiveField))
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
        '''
        filePATH = file_paths_before[i]
        FILE = 'data/'+file_paths_before[i].split('/')[-1]
        print(FILE)
       
        downloadFile(filePATH)
        
        
        # file before
        lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)
      
        sat_h = C['goes_imager_projection'].perspective_point_height
        
        # Satellite longitude
        sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
        
        # Satellite sweep
        sat_sweep = C['goes_imager_projection'].sweep_angle_axis
       
        if i == len(newFileIndexes)-1:
            endIndex = len(GPM_data)
        else:
            endIndex = newFileIndexes[i+1]
        
        
        
        for j in range(newFileIndexes[i],endIndex):
            xIndex, yIndex , distance[j,0]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,1], GPM_data[j,2], sat_h, sat_lon, sat_sweep, x_data,y_data)
            xData[j,0,:,:], times[j,0] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
        '''
        # closest file
        filePATH = filePaths[i]
        #print(filePATH)
        FILE = 'data/'+filePaths[i].split('/')[-1]
        if i % 100 == 0:
            print(i/len(newFileIndexes))
       
        downloadFile(filePATH)
        #filePATH, prev_sat_h = 0, prev_sat_lon = 0, prev_sat_sweep = 0, prev_x = [], prev_y = [], prev_lons =[], prev_lats = []
        #start_time = time.time()
        print(filePath)
        lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)
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
            xIndex, yIndex , distance[j,0]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,0], GPM_data[j,1], proj, x_data,y_data)
            #end_time = time.time()
            #print(" time for getting index %s" %(end_time -start_time))
            
            #start_time = time.time()
            xData[j,0,:,:], times[j,0] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
            #end_time = time.time()
            #print("time for extracting data: %s" %(end_time-start_time))
        #end_time = time.time()
        #print("time for for lop %s" %(end_time-start_time))
        '''
        # file after
        filePATH = file_paths_after[i]
        FILE = 'data/'+file_paths_after[i].split('/')[-1]
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
            xIndex, yIndex , distance[j,2]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,1], GPM_data[j,2], sat_h, sat_lon, sat_sweep, x_data,y_data)
            xData[j,2,:,:], times[j,2] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
        
        '''
        
    
    return xData[:,0,:,:], times[:,0], distance[:,0]   

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
       
def getGPMData(start_DATE, maxDataSize, data_per_GPM_pass, resolution):
    
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
                           str(datetime(2018,5,6)),
                           str(datetime(2019,2,25)),
                           str(datetime(2019,4,15)),
                           str(datetime(2019,5,30)),
                           str(datetime(2019,6,27)),
                           str(datetime(2019,6,28)),
                           str(datetime(2019,11,6))
                           ]
    gpm_data = np.zeros((maxDataSize,4))
   
    
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
            
            
            
            index1 = min(maxDataSize,index+len(precepTotRate),index+data_per_GPM_pass)
        
            
            
            indexes = random.sample(range(0, len(precepTotRate)), index1-index)
       
            
            #print(chosen_prec)
            #data[index:index1,0] = precepTotRate[chosen_indexes]
            #data[index:index1,0] = chosen_prec
            
            '''
            pos_time_data[index:index1,0,:,:] = longitude[chosen_indexes]
            pos_time_data[index:index1,0,:,:] = latitude[chosen_indexes]
            pos_time_data[index:index1,0,:,:] = time[chosen_indexes]
            '''
            
            
            gpm_data[index:index1,0] = longitude[indexes]
            gpm_data[index:index1,1] = latitude[indexes]
            gpm_data[index:index1,2] = time[indexes]
            gpm_data[index:index1,3] = precepTotRate[indexes]
            
            index = index1
            print(index)
            if index >= maxDataSize:
                break
            
           
                
        
        DATE += timedelta(days=1)
        print(DATE)
      
    return gpm_data
def getGPMDataImage(start_DATE, maxDataSize):
    
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
    from scipy.interpolate import griddata
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
    
    pixel_width = 40
    lat_width = 1.8
    lon_width = 1.8
    gpm_data_images = np.zeros((maxDataSize,pixel_width, pixel_width))
    gpm_data_pos_time = np.zeros((maxDataSize,3))
   
    
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
                #dim_1 = f['NS']['surfPrecipTotRate'][:].shape[0]
                #dim_2 = f['NS']['surfPrecipTotRate'][:].shape[1]
                precepTotRate = f['NS']['surfPrecipTotRate'][:]
                longitude = f['NS']['Longitude'][:]
                latitude = f['NS']['Latitude'][:]
                time = f['NS']['navigation']['timeMidScan'][:]
            except:
                continue
            
            # remove null data 
            '''
            indexes = np.where(np.abs(precepTotRate) < 200)
            precepTotRate = precepTotRate[indexes]
            longitude = longitude[indexes]
            latitude = latitude[indexes]
            time = time[indexes]
            '''
            
            import matplotlib.pyplot as plt
            
            
            max_lat = latitude.max()
            
            
            
            #current_index = 0
            #image_height = 35
            #import copy
            #nmb_images = int(dim1/)
            #import copy
            
            #tmp_latitude =copy.deepcopy(latitude) 
            #null_indexes = np.argwhere(tmp_latitude<-5000)
            #tmp_latitude[null_indexes[:,0],null_indexes[:,1]]= 5000
            #tmp_longitude = copy.deepcopy(longitude)
            #null_indexes = np.argwhere(tmp_longitude<-5000)
            #tmp_longitude[null_indexes[:,0],null_indexes[:,1]]= 5000
            
            for i in range(100):
                
                '''
                min_lat = tmp_latitude[current_index:current_index+image_height,:].min()
                max_lat = latitude[current_index:current_index+image_height,:].max()
                min_lon = tmp_longitude[current_index:current_index+image_height,:].min()
                max_lon = longitude[current_index:current_index+image_height,:].max()
                '''
                
                min_lat = max_lat-lat_width
                indexes_lat = np.argwhere((latitude > min_lat) & (latitude < max_lat))
            
                if len(indexes_lat)<2000:
                    break
                    
                max_lon = longitude[indexes_lat[:,0], indexes_lat[:,1]].max()
                min_lon = longitude[indexes_lat[:,0], indexes_lat[:,1]].min()
                
                #print(indexes_lat.shape)
                #print(latitude[indexes_lat[:,0],indexes_lat[:,1]].shape)
                
                #print(min_lat)
                midpoint = [max_lon-np.abs(min_lon-max_lon)/2,max_lat-np.abs(min_lat-max_lat)/2]
                #print(midpoint)
                extent1 = [midpoint[0]-lon_width/2,midpoint[0]+lon_width/2,midpoint[1]-lat_width/2,midpoint[1]+lat_width/2]
                #print(extent1)
                #print(extent1)
                #plt.scatter(latitude[indexes_lat])
                #plt.show()
                #indexes_lat = np.argwhere((latitude[current_index:current_index+image_height,:] > midpoint[1]-lat_width) & (latitude[current_index:current_index+image_height,:] < midpoint[1]+lat_width) )
                indexes = np.argwhere((longitude > extent1[0]) & (longitude < extent1[1]) & (latitude > extent1[2]) & (latitude < extent1[3]) &(precepTotRate> -5000))
                if len(indexes) <1000:
                    max_lat= min_lat
                    continue
                #print(indexes_long.shape)
                #indexes = indexes_long
                #indexes = np.array([value for value in indexes_lat if value in indexes_long])
                # remove null values
                #print(indexes.shape)
                #tmp_non_null = np.where(precepTotRate[indexes[:,0], indexes[:,1]] < 200)[0]
                #print(tmp_non_null.shape)
                #indexes = indexes[tmp_non_null,:]
                
                
                #print(indexes.shape)
                #print(latitude[indexes[:,0],indexes[:,1]].shape)
                points =np.zeros((len(indexes),2)) 
                points[:,0] =longitude[indexes[:,0], indexes[:,1]]
                points[:,1]= latitude[indexes[:,0], indexes[:,1]]
                values = precepTotRate[indexes[:,0], indexes[:,1]]
                #print(values.shape)
                #print(points.shape)
                grid_x, grid_y = np.mgrid[extent1[0]:extent1[1]:40j, extent1[2]:extent1[3]:40j]
                
                grid_z0 = griddata(points,values, (grid_x, grid_y), method='linear', fill_value = 0)
                gpm_data_images[index,:,:] = np.rot90(grid_z0)
                gpm_data_pos_time[index,2] = time[indexes[0,0]]
                gpm_data_pos_time[index,0] = midpoint[0]
                gpm_data_pos_time[index,1] = midpoint[1]
                index+=1
                #plt.scatter(points[:,0], points[:,1], c = values)
                #plt.show()
                #plt.scatter(grid_x.flatten(),grid_y.flatten())
                #plt.show()
                #plt.imshow(np.rot90(grid_z0))
                #plt.show()
                #current_index +=image_height
                #print(midpoint)
                max_lat = min_lat
                if index >= maxDataSize:
                    break
                
            
            
            index1 = min(maxDataSize,index+len(precepTotRate),index+data_per_GPM_pass)
        
            
            
            indexes = random.sample(range(0, len(precepTotRate)), index1-index)
       
            
            #print(chosen_prec)
            #data[index:index1,0] = precepTotRate[chosen_indexes]
            #data[index:index1,0] = chosen_prec
            
            '''
            pos_time_data[index:index1,0,:,:] = longitude[chosen_indexes]
            pos_time_data[index:index1,0,:,:] = latitude[chosen_indexes]
            pos_time_data[index:index1,0,:,:] = time[chosen_indexes]
            '''
            
            
            gpm_data[index:index1,0] = longitude[indexes]
            gpm_data[index:index1,1] = latitude[indexes]
            gpm_data[index:index1,2] = time[indexes]
            gpm_data[index:index1,3] = precepTotRate[indexes]
            
            #index = index1
            print(index)
            if index >= maxDataSize:
                break
            
           
                
        
        DATE += timedelta(days=1)
        print(DATE)
      
    return gpm_data_images, gpm_data_pos_time
   
def convertTimeStampToDatetime(timestamp, time_reference = 'old'):
    from datetime import datetime, timedelta
    if time_reference == 'POSIX':
        return datetime.fromtimestamp(timestamp)
    else:
        return datetime(1980, 1, 6) + timedelta(seconds=timestamp - (35 - 19))
    

def getTrainingData(dataSize, nmb_GPM_pass, GPM_resolution):
    
    import numpy as np
    import datetime
    import time
    
    #receptiveField = 28

    '''
    returns a set that conisit of radiance data for an area around every pixel
    in the given area together with its label
    '''
    
    xData = np.zeros((dataSize,2,receptiveField,receptiveField))
    times = np.zeros((dataSize,3))
    #yData = np.zeros((dataSize,40,40))
    distance = np.zeros((dataSize,2))
    
    
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
    GPM_data = getGPMData(datetime.datetime(2017,8,5),dataSize,nmb_GPM_pass,GPM_resolution)
    times[:,0] = GPM_data[:,2]
    end_time = time.time()
    
    print("time for collecting GPM Data %s" % (end_time-start_time))
    '''
    next step is to pair the label with the geostattionary data.
    '''
    start_time = time.time()
    
    tmp_x, tmp_time, tmp_distance =  getGEOData( GPM_data,'C13')
    xData[:,0,:,:] =tmp_x
    times[:,1] =tmp_time
    distance[:,0] =tmp_distance
    
    #xData[:,1,:,:], times[:,2], distance[:,1] = getGEOData(np.reshape(GPM_pos_time_data, (GPM_pos_time_data.shape[0]*GPM_pos_time_data.shape[2]*GPM_pos_time_data.shape[3],GPM_pos_time_data.shape[1])),'C08')
    tmp_x, tmp_time, tmp_distance =  getGEOData( GPM_data,'C08')
    xData[:,1,:,:] = tmp_x
    times[:,2] = tmp_time
    distance[:,1] = tmp_distance
    
    yData =  GPM_data[:,3]
    end_time = time.time()
    
    print("time for collecting GEO Data %s" % (end_time-start_time))
    import numpy as np
    
    np.save(folder_path+'/trainingData/xDataC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'timeSeries.npy', xData)   
    np.save(folder_path+'/trainingData/yDataC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'timeSeries.npy', yData)   
    np.save(folder_path+'/trainingData/timesC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'timeSeries.npy', times)
    np.save(folder_path+'/trainingData/distanceC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'timeSeries.npy', distance)  
   
    
    return xData, yData, times, distance
def getReferenceDataFileTime(file):
    import datetime
    return (datetime.datetime.strptime(file.split('_')[1].split('.')[0], '%Y%m%d%H%M'))
    
def getReferenceDataLabels(GPM_data):
    from os import listdir
    from os.path import isfile, join
    import numpy as np
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
    
    
    lats = np.arange(lati,latf,DY)
    lons = np.arange(loni,lonf,DX)
    # get label matrix data
    nlin = 1613
    ncol = 1349
    filepath = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\ReferensData\\binaries'
    
    # create list of file name times
    files = listdir(filepath)
    
    # create list of times
    file_times = [getReferenceDataFileTime(file) for file in files]
    
    reference_labels = np.zeros((len(GPM_data),1))
    print(len(GPM_data))
    for i in range(len(GPM_data)):
        
        currentTime = convertTimeStampToDatetime(GPM_data[i,2],'POSIX')
        #print(currentTime)
        #print(currentTime)
        # get the closest file
        time_differences = [np.abs((time-currentTime).total_seconds()) for time in file_times]
        #print(time_differences)
        index = np.argmin(time_differences)
        #print(index)
        current_file = files[index]
        #print(currentTime)
        #print(current_file)
        #print(current_file)
        #print(current_file)
        
        # get teh rain rate array
        rr_array = np.fromfile(filepath+'\\'+current_file, dtype=np.int16,count=nlin*ncol).reshape(nlin, ncol)/10
        y_index = (np.abs(lons-GPM_data[i,0])).argmin()
        x_index = (np.abs(lats-GPM_data[i,1])).argmin()
        #print('GPM')
        #print(GPM_data[i,0])
        #print(GPM_data[i,1])
        #print('hydro')
        #print(lons[y_index])
        #rint(lats[x_index])
        reference_labels[i,0] = rr_array[x_index,y_index]
        if i % 500 ==0:    
            print(i)

    
    return reference_labels[:,0]
def getReferenceData():
    import numpy as np
    import datetime
    import time
    
    #receptiveField = 28

    '''
    returns the gpm data for the availiablle reference data along with the
    geostationary window and the referens label
    '''
    dataSize = 30000
    nmb_GPM_pass = 10000
    GPM_resolution = 1
    xData = np.zeros((dataSize,2,receptiveField,receptiveField))
    times = np.zeros((dataSize,3))
    yData = np.zeros((dataSize,2))
    distance = np.zeros((dataSize,2))
    
    
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
    GPM_data= getGPMData(datetime.datetime(2019,3,20),dataSize,nmb_GPM_pass,GPM_resolution)
    #print(convertTimeStampToDatetime(GPM_data[-1,2],'old'))
    times[:,0] = GPM_data[:,2]
    
    end_time = time.time()
    
    print("time for collecting GPM Data %s" % (end_time-start_time))
    '''
    next step is to pair the label with the geostattionary data.
    '''
    start_time = time.time()
    
    tmp_x, tmp_time, tmp_distance =  getGEOData(GPM_data,'C13')
    xData[:,0,:,:] =tmp_x
    times[:,1] =tmp_time
    distance[:,0] =tmp_distance
    
    #xData[:,1,:,:], times[:,2], distance[:,1] = getGEOData(np.reshape(GPM_pos_time_data, (GPM_pos_time_data.shape[0]*GPM_pos_time_data.shape[2]*GPM_pos_time_data.shape[3],GPM_pos_time_data.shape[1])),'C08')
    tmp_x, tmp_time, tmp_distance =  getGEOData(GPM_data,'C08')
    xData[:,1,:,:] = tmp_x
    times[:,2] = tmp_time
    distance[:,1] = tmp_distance
    
    yData[:,0] = GPM_data[:,3]
    end_time = time.time()
    
    print("time for collecting GEO Data %s" % (end_time-start_time))
    yData[:,1] = getReferenceDataLabels(GPM_data)
    
    import numpy as np
    
    np.save(folder_path+'/trainingData/xDataC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'referenceSingle.npy', xData)   
    np.save(folder_path+'/trainingData/yDataC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'referenceSingle.npy', yData)   
    np.save(folder_path+'/trainingData/timesC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'referenceSingle.npy', times)
    np.save(folder_path+'/trainingData/distanceC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'referenceSingle.npy', distance)  
    np.save(folder_path+'/trainingData/positionC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'referenceSingle.npy', GPM_data)  
    
    
    return xData, yData, times, distance
    
    

def load_gauge_data():
    from os import listdir
    from os.path import isfile, join
    from datetime import datetime
    import numpy as np
    import time
    
    file_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\ReferensData\\pluvi'
    
    # get all the gauge data file names
    file_names = listdir(file_path)
    line_count = 42960
    gauge_data = np.zeros((line_count,4))
    line_number = 0
    for file in file_names:
        
        date = datetime.strptime(file[4:-4], '%Y%m%d%H')
        #print(date)
        #print(file)
        with open(file_path +'\\'+ file) as f:
            for cnt, line in enumerate(f):
                values = line.split('  ')
                #print(date.timestamp())
                #print(values)
                #print(type(values[1]))
                #print(type(values[2]))
                #print(type(values[-1]))
                #break
                #if isinstance(values[1], float) and isinstance(values[2], float) and isinstance(values[-1], float):
                #print(values[-1][:-2])
                try:
                    gauge_data[line_number,:] = [values[1],values[2],date.timestamp(),values[-1]]
                except ValueError:
                    try:
                        #print(values)
                        gauge_data[line_number,:] = [values[1],values[3],date.timestamp(),values[-1]]
                    except ValueError:
                        #print(values)
                        pass
                    pass
                #elif isinstance(values[1], float) and isinstance(values[3], float) and isinstance(values[-1], float):
                #    data[line_number,:] = [values[1],values[3],values[-1],date.timestamp()]
                #print(gauge_data[line_number,-1])
                line_number +=1
                
    #print(line_number)
    #import matplotlib.pyplot as plt
    
    indexes = np.where((gauge_data[:,0] > minLongitde) &
                   (gauge_data[:,0] < maxLongitude) &
                   (gauge_data[:,1] > minLatitude) &
                   (gauge_data[:,1] < maxLatitide) &
                   (gauge_data[:,-1] < 8000))
    
    #print(line_number)
    #print(indexes.shape)
    #print(len(indexes[0]))
    gauge_data = gauge_data[indexes,:][0,:]
    #plt.plot(gauge_data[:,-1])
    test_data = np.zeros((gauge_data.shape[0],3,5))
    for i in  range(gauge_data.shape[0]):
        test_data[i,0,:-1] = gauge_data[i,:]
        test_data[i,1,:-1] = gauge_data[i,:]
        test_data[i,2,:-1] = gauge_data[i,:]
        
        test_data[i,0,2] = gauge_data[i,2] - 15*60
        test_data[i,1,2] = gauge_data[i,2] - 30*60
        test_data[i,2,2] = gauge_data[i,2] - 45*60
        
        test_data[i,0,-1] = i
        test_data[i,1,-1] = i
        test_data[i,2,-1] = i
    
        
    test_data = np.reshape(test_data, (test_data.shape[0]*3,5))
    test_data = test_data[test_data[:,2].argsort()]
    dataSize = gauge_data.shape[0]*3
    tmpxData = np.zeros((dataSize,2,receptiveField,receptiveField))
    tmptimes = np.zeros((dataSize,3))
    #yData = np.zeros((dataSize,40,40))
    tmpdistance = np.zeros((dataSize,2))
    
    tmp_hydro = getReferenceDataLabels(test_data)
    #print(tmp_hydro.shape)
    hydro = np.zeros((len(gauge_data),1))
    for i in range(len(gauge_data)):
        indexes = np.where(test_data[:,-1] == i)[0]
        #print(indexes)
        values = np.sort(tmp_hydro[indexes])
        
        hydro[i] = (values[0]+2*values[1]+values[2])/4
        print(i)
    #tmp_hydro = np.reshape(tmp_hydro, (len(gauge_data),4))
    #print(tmp_hydro.shape)
    #hydro = np.mean(tmp_hydro, axis = 1)

    
    tmp_x, tmp_time, tmp_distance =  getGEOData(test_data,'C13','POSIX')
    tmpxData[:,0,:,:] =tmp_x
    tmptimes[:,1] =tmp_time
    tmpdistance[:,0] =tmp_distance
    
    #xData[:,1,:,:], times[:,2], distance[:,1] = getGEOData(np.reshape(GPM_pos_time_data, (GPM_pos_time_data.shape[0]*GPM_pos_time_data.shape[2]*GPM_pos_time_data.shape[3],GPM_pos_time_data.shape[1])),'C08')
    tmp_x, tmp_time, tmp_distance =  getGEOData(test_data,'C08', 'POSIX')
    tmpxData[:,1,:,:] = tmp_x
    tmptimes[:,2] = tmp_time
    tmpdistance[:,1] = tmp_distance
    
    xData = np.zeros((gauge_data.shape[0],3,2,28,28))
    times = np.zeros((gauge_data.shape[0],3,3))
    #yData = np.zeros((dataSize,40,40))
    distance = np.zeros((gauge_data.shape[0],3,2))
    
    for i in range(len(gauge_data)):
        indexes = np.where(test_data[:,-1] == i)[0]
        xData[i,:] = tmpxData[indexes,:]
        times[i,:] = tmptimes[indexes,:]
        distance[i,:] = tmpdistance[indexes,:]
        print(i)
    #xData = np.reshape(xData, (gauge_data.shape[0],4,2,28,28))
    #times = np.reshape(times, (gauge_data.shape[0],4,3))
    #distance = np.reshape(distance, (gauge_data.shape[0],4,2))
    
    np.save(folder_path+'/trainingData/xDataC8C13SreferenceTrim3Values.npy', xData)   
    np.save(folder_path+'/trainingData/timesC8C13SreferenceTrim3Values.npy', times)
    np.save(folder_path+'/trainingData/distanceC8C13SreferenceTrim3Values.npy', distance)  
    np.save(folder_path+'/trainingData/hydropredC8C13SreferenceTrim3Values.npy', hydro)  
    np.save(folder_path+'/trainingData/gaugeC8C13SreferenceTrim3Values.npy', gauge_data)  
    
    return hydro,gauge_data, xData, times, distance
    
    
def plotTrainingData(xData,yData, times, nmbImages):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(5800,5800+nmbImages):
        
        fig = plt.figure()
        fig.suptitle('timediff %s, rainfall %s' % (np.abs(times[i,0]-times[i,1]), yData[i]), fontsize=20)
        plt.imshow(xData[i,:,:])
        
def extract_data_within_timewindow(xData, yData, times, distance):

    indexes = np.where(np.abs(times[:,0]-times[:,1]) <200)
    return xData[indexes[0],:,:] , yData[indexes[0]] , times[indexes[0],:], distance[indexes[0],:]

def preprocessDataForTraining(xData, yData, times, distance):

    import numpy as np
   

    # reshape data for the QRNN
    newXData = np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]))
    #newYData = np.reshape(yData,(len(yData),1))
    
    #scaler1.fit(newXData)
    #newXData = scaler1.transform(newXData)
    #newYData = newYData/newYData.max()
    
    # comine the IR images and the distance and time difference
    tmp = np.zeros((xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]+4))
    tmp[:,:xData.shape[1]*xData.shape[2]*xData.shape[3]] = newXData
    
    '''
    tmp[:,-1] = (times[:,0]-times[:,1])/(times[:,0]-times[:,1]).max()
    tmp[:,-2] = distance[:,0]/distance.max()
    tmp[:,-3] = (times[:,0]-times[:,2])/(times[:,0]-times[:,2]).max()
    tmp[:,-4] = distance[:,1]/distance.max()
    '''
    
    tmp[:,-1] = (times[:,0]-times[:,1])/1000
    tmp[:,-2] = distance[:,0]
    tmp[:,-3] = distance[:,1]
    tmp[:,-4] = (times[:,0]-times[:,2])/1000
    
  
    
    newXData = tmp
    
    # scale the data with unit variance and and between 0 and 1 for the labels
    
    return newXData
def get_single_GPM_pass(DATE): 
    import numpy as np
    import h5py
    
    files = getGPMFilesForSpecificDay(DATE)
   # FILENAME = files[0]
    
    # download the gpm data
   # downloadGPMFile(files[0],DATE)
    
    # read the data
    
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
                indexes = np.where(np.abs(precepTotRate) < 200)[0]
                precepTotRate = precepTotRate[indexes]
                longitude = longitude[indexes]
                latitude = latitude[indexes]
                time = time[indexes]
                return precepTotRate, longitude, latitude, time
            except:
                continue
            # remove null data 
            

def load_data(set_nmb):
    import numpy as np
    
    if set_nmb == 1:
        
        xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
        yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
        times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
        distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
        #xData = xData[:,:,6:22,6:22]
    
    '''
    xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
    yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
    times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
    distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
    #xData = xData[:,:,6:22,6:22]
    '''
    '''
    xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
    yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
    times = np.load(folder_path+'/trainingData/timesC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
    distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R128_P1400GPM_res3timeSeries.npy') 
    '''
    if set_nmb == 3:
        
        xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
        yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R100_P1400GPM_res3interval_3.npy')
        times = np.load(folder_path+'/trainingData/timesC8C13S6200_R100_P1400GPM_res3interval_3.npy')
        distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R100_P1400GPM_res3interval_3.npy') 
        
    if set_nmb == 2:
    
        xData =np.load(folder_path+'/trainingData/xDataC8C13S320000_R28_P200GPM_res3timeSeries.npy')
        yData = np.load(folder_path+'/trainingData/yDataC8C13S320000_R28_P200GPM_res3timeSeries.npy')
        times = np.load(folder_path+'/trainingData/timesC8C13S320000_R28_P200GPM_res3timeSeries.npy')
        distance = np.load(folder_path+'/trainingData/distanceC8C13S320000_R28_P200GPM_res3timeSeries.npy') 
        
    
    '''
    xData =np.load('trainingData/xDataC8C13S3200_R28_P4GPM_res3.npy')
    yData = np.load('trainingData/xDataC8C13S3200_R28_P4GPM_res3.npy')
    times = np.load('trainingData/xDataC8C13S3200_R28_P4GPM_res3.npy')
    distance = np.load('trainingData/xDataC8C13S3200_R28_P4GPM_res3.npy')  
    '''
    '''
    xData =np.load(folder_path+'/trainingData/xDataC8C13S30000_R28_P10000GPM_res1referenceSingle.npy')
    yData = np.load(folder_path+'/trainingData/yDataC8C13S30000_R28_P10000GPM_res1referenceSingle.npy')
    times = np.load(folder_path+'/trainingData/timesC8C13S30000_R28_P10000GPM_res1referenceSingle.npy')
    distance = np.load(folder_path+'/trainingData/distanceC8C13S30000_R28_P10000GPM_res1referenceSingle.npy') 
    
    # remove nan values
    scalenanValues =np.argwhere(np.isnan(scalexData)) 
    scalexData = np.delete(scalexData,np.unique(scalenanValues[:,0]),0)
    '''
    
    #xData = xData[:,:,14:114,14:114]
    nanValues =np.argwhere(np.isnan(xData)) 
    xData = np.delete(xData,np.unique(nanValues[:,0]),0)
    yData = np.delete(yData,np.unique(nanValues[:,0]),0)
    times = np.delete(times,np.unique(nanValues[:,0]),0)
    distance = np.delete(distance,np.unique(nanValues[:,0]),0)
    '''
    indexes = np.where((np.abs(times[:,0]-times[:,1])) < 200)[0]

    xData = xData[indexes,:]
    yData = yData[indexes,:]
    times = times[indexes,:]
    distance = distance[indexes,:]
    '''
    #print(xData.shape)
    #print(yData.shape)
    # get the mean of the yData
    #tmp_yData = np.zeros((len(yData),1))
    #for i in range(len(yData)):
    #    tmp_yData[i,0] = yData[i]
    #yData = np.reshape(yData[:,3,3], (len(yData),1))
    # narrow the field of vision
    #xData = xData[:,:,9:19,9:19]
    
    # select the data within time limit
    '''
    indexes = np.where(np.abs(times[:,0]-times[:,1])<200)[0]
    xData = xData[indexes,:,:,:]
    yData = yData[indexes]
    times = times[indexes]
    distance = distance[indexes]
    '''
    #np.mean(xTrain, axis=0, keepdims=True)
   
    
    '''
    tmpXData = np.zeros((len(xData),6,28,28,1))
    tmpXData[:,0,:,:,0] = xData[:,0,:,:]
    tmpXData[:,1,:,:,0] = xData[:,3,:,:]
    tmpXData[:,2,:,:,0] = xData[:,1,:,:]
    tmpXData[:,3,:,:,0] = xData[:,4,:,:]
    tmpXData[:,4,:,:,0] = xData[:,2,:,:]
    tmpXData[:,5,:,:,0] = xData[:,5,:,:]
    '''

    #tmpXData = np.zeros((len(xData),100,100,2))
    #tmpXData = np.zeros((len(xData),28,28,2))
    
    if set_nmb == 2:
        tmpXData = np.zeros((1,28,28,2))
    elif set_nmb == 1:
        tmpXData = np.zeros((len(xData),28,28,2))
    elif set_nmb == 3:
        tmpXData = np.zeros((len(xData),28,28,2))
    
    for i in range(2):
        #mean1 = np.mean(xData[:,i,:,:])
        #std1 = np.std(xData[:,i,:,:])
        #xData[:,i,:,:] =  (xData[:,i,:,:]-mean1)/std1
    
        tmpXData[:,:,:,i] = (xData[:,i,:,:]-xData[:,i,:,:].min())/(xData[:,i,:,:].max()-xData[:,i,:,:].min())
    
    
    '''
    for i in range(6):
        #tmpXData[:,:,:,i] = (xData[:,i,:,:]-mean1)/std1
    
        tmpXData[:,i,:,:,:] = (tmpXData[:,i,:,:,:]-tmpXData[:,i,:,:,:].min())/(tmpXData[:,i,:,:,:].max()-tmpXData[:,i,:,:,:].min())
    '''
    '''
    '''
    
    #tmpXData = preprocessDataForTraining(xData, yData, times, distance)
        
    #xData = tmpXData
    # yData = (yData-np.mean(yData))/np.std(yData)
    # yData = np.sqrt(yData)
    #newXData = preprocessDataForTraining(xData, yData, times, distance)
    
    #tmpXData = np.zeros((len(xData),xData.shape[2],xData.shape[3],xData.shape[1]))
    '''
    for i in range(6):
 
            tmpXData[:,i,:,:,:] = (tmpXData[:,i,:,:,:]-tmpXData[:,i,:,:,:].min())/(tmpXData[:,i,:,:,:].max()-tmpXData[:,i,:,:,:].min()) 
    '''
    return tmpXData, yData
    
def preprocessData(xData, yData, model, scale):
    import numpy as np
    '''
    INPUT: xData has dimensions (nmb_samples,channels, image_height, image_width)
    '''
    tmpXData = np.zeros((len(xData),xData.shape[2],xData.shape[3],xData.shape[1]))
    if model =='CNN':
        for i in range(xData.shape[1]):
            if scale =='01':
                tmpXData[:,:,:,i] = (xData[:,i,:,:]-xData[:,i,:,:].min())/(xData[:,i,:,:].max()-xData[:,i,:,:].min()) 
                
    return tmpXData, yData