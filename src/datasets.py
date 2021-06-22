# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:02:43 2020

@author: gustav
"""

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class GOES_GPM_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, xData, yData, x_mean, x_sigma, device):
        
        self.device = device
        #self.xData =  torch.tensor(np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3])),dtype=torch.float, requires_grad = True ).flatten()
        #self.yData = torch.tensor( np.reshape(yData,(len(yData),)),dtype=torch.float)
        #self.xData =  
        #self.yData =  np.reshape(yData,(len(yData),))
        #self.xData = torch.from_numpy(np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]))).float()
        #self.yData = torch.from_numpy(np.reshape(yData,(len(yData),))).float()
        self.xData = torch.from_numpy(xData).float()
        self.yData = torch.from_numpy(np.reshape(yData,(len(yData),))).float()
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        
        

    def __len__(self):
        return self.xData.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.xData.shape)
        inputs =(self.xData[idx,:]-self.x_mean)/self.x_sigma
        labels =self.yData[idx]
        #labels = labels.to(device)
        #inputs = inputs.to(device)
        

        #if self.transform:
        #    sample = self.transform(sample)

        return inputs, labels