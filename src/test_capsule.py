import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsulenet import CapsNet


USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
N_EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9

'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self, dataset='mnist'):
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'GOES':
            # CNN (cnn)
            self.cnn_in_channels = 2
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 5
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28


def train(net,optimizer, trainloader,device,tau,epoch):
    net.train()
    running_loss = 0.0
    #net.to(device)
    #suffle_indexes = random.sample(range(0, x_train.shape[0]), x_train.shape[0])
    #x_train = x_train[suffle_indexes,:]
    #y_train = y_train[suffle_indexes]
    #for i in range(2):
    #for i in range(int(len(x_train)/batch_size)):
    
    i =0
    for inputs, labels in trainloader:
        #print(i)
        # get the inputs; data is a list of [inputs, labels]
        #batch_indexes = list(range(i*batch_size,(i+1)*batch_size))
        #print(batch_indexes)
        #inputs = torch.tensor(np.reshape(x_train[batch_indexes,:],(batch_size,2,28,28)),dtype=torch.float, requires_grad = True)
        
        #inputs = torch.tensor(x_train[batch_indexes,:],dtype=torch.float, requires_grad = True )
        #labels =torch.tensor( np.reshape(y_train[batch_indexes],(batch_size,)),dtype=torch.float,requires_grad = True)
        #inputs = x_train[batch_indexes,:]
        #labels = y_train[batch_indexes]
        labels = labels.to(device)
        inputs = inputs.to(device)
        
        #labels.cuda()
        #inputs.cuda()
        
        #inputs = data['inputs']
        #labels = data['labels']
        #print(inputs)
        #print(device)
        #inputs.requires_grad = True
        #inputs = torch.tensor(inputs,dtype=torch.float, requires_grad = True )
        #labels = torch.tensor(labels,dtype=torch.float, requires_grad = True )
        #inputs = inputs.to(device)
        #labels = labels.to(device)
        #inputs.cuda()
        #labels.cuda()
        #print(inputs.device)
        #print(data['inputs'])
        

        # forward + backward + optimize
        output, reconstructions, masked = capsule_net(inputs)
        #outputs = net(inputs)
        #print(outputs.shape)
        #loss = criterion(outputs, labels)
        loss = capsule_net.loss(inputs, output, labels, reconstructions)
        #loss = quantile_loss(outputs, labels, tau)
        #print(loss)
        loss.backward()
        
        
        optimizer.step()
        optimizer.zero_grad()
        #labels = labels.detach()
        #inputs = inputs.detach()
        # scheduler.step()
        # print statistics
        running_loss += loss.item()
        i+=1
        
   
    val_loss = 0
    print('[%d, %5d] loss: %.3f val_loss: %3f' %(epoch, i, running_loss/i, val_loss))    
    running_loss = 0.0
    
def test(net, tau, validation_loader, device,e):
     
    net.eval()
    n = 0
    tot_val_loss = 0
    #net.train(False)
    with torch.no_grad():
        for inputs, labels in validation_loader:
            #torch.cuda.empty_cache()
            #val_input = torch.tensor(np.reshape(x_test[j*n:(j+1)*n,:],(len(x_test[j*n:(j+1)*n,:]),2,28,28)),dtype=torch.float,requires_grad = True) 
            
            #val_input = torch.tensor(x_test[j*n:(j+1)*n,:],dtype=torch.float,requires_grad = True) 
            #val_labels = torch.tensor( np.reshape(y_test[j*n:(j+1)*n],(len(y_test[j*n:(j+1)*n]),)),dtype=torch.float)
            labels = labels.to(device)
            inputs = inputs.to(device)
            
            #val_labels = val_labels.to(device)
            #val_input = val_input.to(device)
            
            output, reconstructions, masked = capsule_net(inputs)
            #outputs = net(inputs)
            #print(outputs.shape)
            #loss = criterion(outputs, labels)
            val_loss = capsule_net.loss(inputs, output, labels, reconstructions)
            #print(val_loss.cpu().data.numpy())
            tot_val_loss = tot_val_loss+val_loss.cpu().data.numpy()
            n+=1
        
    #net.train(True)
    val_loss = tot_val_loss/n
    print(val_loss)
   
#%%

import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms




# load the data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#xDataC8C13S130000_R28_P200_R0.5.npy

xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  

# remove nan values
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)



# get the mean of the yData
tmp_yData = np.zeros((len(yData),1))
for i in range(len(yData)):
    tmp_yData[i,0] = yData[i,3,3]

yData = tmp_yData/tmp_yData.max()    
#xData = xData[:,0,:,:]+xData[:,1,:,:]
#xData = xData[:,:,10:18,10:18]
print(xData.shape)
#xData = np.reshape(xData, (len(xData),2*28*28))

#scaler1 = StandardScaler()
#scaler1.fit(xData)
#xData = scaler1.transform(xData)
#xData = preprocessing.normalize(xData, norm='l2')

xData[:,0,:,:] = (xData[:,0,:,:]-np.mean(xData[:,0,:,:]))/np.std(xData[:,0,:,:])
xData[:,1,:,:] =(xData[:,1,:,:]-np.mean(xData[:,1,:,:]))/np.std(xData[:,1,:,:])

#xData=(xData-np.mean(xData))/np.std(xData)

#xData = np.reshape(xData, (len(xData),2*8*8))

#yData = yData/yData.max()


split_index = 175000

x_train = xData[:split_index,:]
x_test = xData[split_index:,:]

y_train = yData[:split_index]
y_test = yData[split_index:]
#%%
x_mean = np.mean(x_train)
x_sigma = np.std(x_train)

#%%
import random
indexes = random.sample(range(0,len(x_train)),1000)

tmp_x_train = x_train[indexes,:]
tmp_y_train = y_train[indexes,:]


indexes = random.sample(range(0,len(x_test)),1000)

tmp_x_test = x_test[indexes,:]
tmp_y_test = y_test[indexes,:]
#%%
import random
from torch.optim.lr_scheduler import StepLR
from datasets import GOES_GPM_Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
dataset = 'GOES'
# dataset = 'mnist'
config = Config(dataset)
#mnist = Dataset(dataset, BATCH_SIZE)

capsule_net = CapsNet(config)
capsule_net = torch.nn.DataParallel(capsule_net)
if USE_CUDA:
    capsule_net = capsule_net.cuda()
capsule_net = capsule_net.module

optimizer = torch.optim.Adam(capsule_net.parameters())
trainset = GOES_GPM_Dataset(x_train,y_train,x_mean, x_sigma, device)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True)
testset = GOES_GPM_Dataset(x_test,y_test,x_mean, x_sigma, device)

test_loader = torch.utils.data.DataLoader(testset, batch_size=256,
                                          shuffle=False)

tau = [0.1,0.3,0.5,0.7,0.9]
for e in range(1, N_EPOCHS + 1):
    train(capsule_net,optimizer, train_loader,device,tau,e)
    test(capsule_net,tau, test_loader,device,e)
#%%
net = capsule_net
net.eval()
n = 0
tot_val_loss = 0
#net.train(False)
test_output = np.zeros((len(x_test),5))
with torch.no_grad():
    for inputs, labels in test_loader:
        #torch.cuda.empty_cache()
        #val_input = torch.tensor(np.reshape(x_test[j*n:(j+1)*n,:],(len(x_test[j*n:(j+1)*n,:]),2,28,28)),dtype=torch.float,requires_grad = True) 
        
        #val_input = torch.tensor(x_test[j*n:(j+1)*n,:],dtype=torch.float,requires_grad = True) 
        #val_labels = torch.tensor( np.reshape(y_test[j*n:(j+1)*n],(len(y_test[j*n:(j+1)*n]),)),dtype=torch.float)
        labels = labels.to(device)
        inputs = inputs.to(device)
        
        #val_labels = val_labels.to(device)
        #val_input = val_input.to(device)
        
        output, reconstructions, masked = capsule_net(inputs)
        test_output[n*256:(n+1)*256,:]=torch.sqrt((output ** 2).sum(dim=2, keepdim=True)).cpu().data.numpy()[:,:,0,0]
        #outputs = net(inputs)
        #print(outputs.shape)
        #loss = criterion(outputs, labels)
        #val_loss = capsule_net.loss(inputs, output, labels, reconstructions)
        #print(val_loss.cpu().data.numpy())
        #tot_val_loss = tot_val_loss+val_loss.cpu().data.numpy()
        n+=1
    
#net.train(True)
#val_loss = tot_val_loss/n
#print(val_loss)

#%%
tmp_output = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))
tmp_output = tmp_output.cpu().data.numpy()
tmp_labels = labels.cpu().data.numpy()
#%%
quantiles = [0.1,0.3,0.5,0.7,0.9]
from visulize_results import generate_all_results_CNN
generate_all_results_CNN(test_output,np.reshape(test_output[:,2],(len(test_output[:,2]),1)),x_test, y_test,y_train, quantiles)
    
