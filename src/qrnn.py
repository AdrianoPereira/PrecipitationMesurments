# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:18:57 2020

@author: gustav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

nmb_cannels = 2
input_width = 28
class QRNN(nn.Module):
    '''
    QRNN implementation in pytorch. Based on the kears implementation found at 
    https://github.com/atmtools/typhon/tree/master/typhon/retrieval/qrnn.
    '''

    def __init__(self, input_dim):
        super(QRNN, self).__init__()
        
        
        # an affine operation: y = Wx + b
        
        self.conv1 = nn.Conv2d(2, 20, 2, padding = 1)
        self.conv2 = nn.Conv2d(20, 50, 2, padding = 1)
        #self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2*2*50, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        #self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(500, 500)
        self.bn3 = nn.BatchNorm1d(500)
        #self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(500, 5)   
        '''
    
        dim = 256
        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        # self.dropout1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc4 = nn.Linear(dim, dim)
        self.bn3 = nn.BatchNorm1d(dim)
        #self.dropout2 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(dim, dim)
        self.bn4 = nn.BatchNorm1d(dim)
        self.fc6 = nn.Linear(dim, dim)
        self.bn5 = nn.BatchNorm1d(dim)
        self.fc7 = nn.Linear(dim, dim)
        self.bn6 = nn.BatchNorm1d(dim)
        self.fc8 = nn.Linear(dim, dim) 
        self.fc9 = nn.Linear(dim ,5) 
        '''
    def forward(self, x):
        
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #x = self.dropout1(x)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        #x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        #x = self.dropout3(x)
        x = self.fc4(x)
        '''
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.bn1(x)
        #x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = self.bn2(x)
        x = F.relu(self.fc4(x))
        x = self.bn3(x)
        #x = self.dropout2(x)
        x = F.relu(self.fc5(x))
        x = self.bn4(x)
        x = F.relu(self.fc6(x))
        x = self.bn5(x)
        x = F.relu(self.fc7(x))
        x = self.bn6(x)
        x = F.relu(self.fc8(x))
        
        x = self.fc9(x)
        '''
        return x
    
   

#%%
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import GOES_GPM_Dataset




# load the data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#xDataC8C13S130000_R28_P200_R0.5.npy

xData =np.load('trainingData/xDataC8C13S200000_R28_P200.npy')
yData = np.load('trainingData/yDataC8C13S200000_R28_P200.npy')
times = np.load('trainingData/timesC8C13S200000_R28_P200.npy')
distance = np.load('trainingData/distanceC8C13S200000_R28_P200.npy')  



print(xData.shape)
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)

#xData = xData[:,0,:,:]+xData[:,1,:,:]
xData = xData[:,:,10:18,10:18]
print(xData.shape)
#xData = np.reshape(xData, (len(xData),2*28*28))

#scaler1 = StandardScaler()
#scaler1.fit(xData)
#xData = scaler1.transform(xData)
#xData = preprocessing.normalize(xData, norm='l2')

xData[:,0,:,:] = (xData[:,0,:,:]-np.mean(xData[:,0,:,:]))/np.std(xData[:,0,:,:])
xData[:,1,:,:] =(xData[:,1,:,:]-np.mean(xData[:,1,:,:]))/np.std(xData[:,1,:,:])
#xData = np.reshape(xData, (len(xData),2*8*8))
    
#yData = yData/yData.max()


split_index = int(len(xData)*0.75)

x_train = xData[:split_index,:]
x_test = xData[split_index:,:]

y_train = yData[:split_index]
y_test = yData[split_index:]


#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = QRNN(8*8*2)
net.to(device)
def skewed_absolute_error(output, target, tau):
    """
    The quantile loss function for a given quantile tau:
    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)
    Where I is the indicator function.
    """
    dy = output - target
    #print(dy.size())
    return torch.mean((1.0 - tau) * torch.relu(dy) + tau * torch.relu(-dy), axis=-1)


def quantile_loss(output, target, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    e = skewed_absolute_error(
        torch.flatten(target), torch.flatten(output[:, 0]), taus[0])
    #print(e)
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(torch.flatten(output[:, i + 1]),
                                   torch.flatten(target),
                                   tau)
    return e

# define loss and optimizer
import random
optimizer = optim.SGD(net.parameters(), lr=0.01) 
batch_size =128
tau = [0.1,0.3,0.5,0.7,0.9]
net.train(True)
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    
    suffle_indexes = random.sample(range(0, x_train.shape[0]), x_train.shape[0])
    x_train = x_train[suffle_indexes,:]
    y_train = y_train[suffle_indexes]
    #for i in range(2):
    for i in range(int(len(x_train)/batch_size)):
        # get the inputs; data is a list of [inputs, labels]
        batch_indexes = list(range(i*batch_size,(i+1)*batch_size))
        #inputs = torch.tensor(np.reshape(x_train[batch_indexes,:],(batch_size,2,28,28)),dtype=torch.float, requires_grad = True)
        
        inputs = torch.tensor(x_train[batch_indexes,:],dtype=torch.float, requires_grad = True)
        labels =torch.tensor( np.reshape(y_train[batch_indexes],(batch_size,)),dtype=torch.float)   
        labels = labels.to(device)
        inputs = inputs.to(device)
        
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        #loss = criterion(outputs, labels)
        loss = quantile_loss(outputs, labels, tau)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    '''
    with torch.no_grad():
        val_input = torch.tensor(np.reshape(x_test,(len(x_test),2,28,28)),dtype=torch.float,requires_grad = True) 
        val_labels = torch.tensor( np.reshape(y_test,(len(y_test),)),dtype=torch.float) 
        val_labels = val_labels.to(device)
        val_input = val_input.to(device)
        val_loss = quantile_loss(net(val_input),val_labels,tau)
        #val_loss = criterion(net(val_input),val_labels)
    '''
    n = 500
    tot_val_loss = 0
    net.train(False)
    with torch.no_grad():
        for j in range(int(len(x_test)/n)):
            #torch.cuda.empty_cache()
            #val_input = torch.tensor(np.reshape(x_test[j*n:(j+1)*n,:],(len(x_test[j*n:(j+1)*n,:]),2,28,28)),dtype=torch.float,requires_grad = True) 
            
            val_input = torch.tensor(x_test[j*n:(j+1)*n,:],dtype=torch.float,requires_grad = True) 
            val_labels = torch.tensor( np.reshape(y_test[j*n:(j+1)*n],(len(y_test[j*n:(j+1)*n]),)),dtype=torch.float) 
            val_labels = val_labels.to(device)
            val_input = val_input.to(device)
            val_loss = quantile_loss(net(val_input),val_labels,tau)
            #print(val_loss.cpu().data.numpy())
            tot_val_loss = tot_val_loss+val_loss.cpu().data.numpy()
        
    net.train(True)
    val_loss = tot_val_loss/int(len(x_test)/n)
    print('[%d, %5d] loss: %.3f val_loss: %3f' %(epoch + 1, i + 1, running_loss/int(len(x_train)/batch_size), val_loss))    
    running_loss = 0.0
    
    

print('Finished Training')

#%% make predictions
random.sample(range(0, len(x_test)), len(x_test))
n = 500
tot_val_loss = 0
out = np.zeros((len(x_test),5))
for i in range(int(len(x_test)/n)):
    #torch.cuda.empty_cache()
    val_input = torch.tensor(x_test[i*n:(i+1)*n,:],dtype=torch.float,requires_grad = True) 
    #val_labels = torch.tensor( np.reshape(y_test[i*n:(i+1)*n],(len(y_test[i*n:(i+1)*n]),)),dtype=torch.float) 
    #val_labels = val_labels.to(device)
    val_input = val_input.to(device)
    out[i*n:(i+1)*n,:] = net(val_input).cpu().data.numpy()
    #val_loss = quantile_loss(net(val_input),val_labels,tau)
    #print(val_loss.cpu().data.numpy())
    #tot_val_loss = tot_val_loss+val_loss.cpu().data.numpy()
    
print(tot_val_loss/int(len(x_test)/n))
#val_loss = criterion(net(val_input),val_labels)

#%% enterpret the results
output = out
import matplotlib.pyplot as plt

plt.plot(output[:,4])
plt.plot(y_test, alpha = 0.5)
plt.show()
#%%
from visulize_results import generateQQPLot
print(y_test.shape)
generateQQPLot(tau,np.reshape(y_test,(len(y_test),1)), output)
#%%
from visulize_results import generate_qqplot_for_intervals
generate_qqplot_for_intervals(tau, y_test[:20000], np.flip(output.cpu().detach().numpy()), 1)
#%%
from visulize_results import confusionMatrix
print(len(np.where(y_test > 0)[0]))
confusionMatrix(y_test,output[:,4])


#%%print(type(testset))
import numpy as np
xData =np.load('trainingData/xDataC8C13S500000_R28_P1000_R0.5.npy')
yData = np.load('trainingData/yDataC8C13S500000_R28_P1000_R0.5.npy')
times = np.load('trainingData/timesC8C13S500000_R28_P1000_R0.5.npy')
distance = np.load('trainingData/distanceC8C13S500000_R28_P1000_R0.5.npy')

