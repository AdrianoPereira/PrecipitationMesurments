# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:02:08 2020

@author: gustav
This is example code for how to create datasets of GOES 16 Infrared images paired with 
GPM rain rates in different constellations

The main idea of the algorithm is as follows:
    1: sample data from each GPM pass starting at date START_DATE.
    2: from each pass, take a random sample of size SAMPLE_SIZE_FROM_EACH_PASS
    3: the GOES data is then paired by selecting the image centered at the GPM data.
    4: this is done until the desired dataset size is reached.
    
"""


'''
    Example 1:  GOES image size: 28x28 pixels
                GPM label: 1 pixel
                Total data size: 350 000
                Number of data per GPM pass: 200
                
'''
from creating_datasets import create_dataset
from datetime import datetime

data_size = 1000
samplesize_per_pass = 200
GOES_image_width = 28
window = [-70, -51, -11, 2.5]
folder_path = 'E:/Precipitation_mesurments'
START_DATE = datetime(2017,8,5)

xData, yData , times, distance = create_dataset(data_size, 
                                               samplesize_per_pass,
                                               GOES_image_width,
                                               START_DATE,
                                               window,
                                               folder_path)



