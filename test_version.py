import numpy as np
from data_loader import load_data
import random
import extendedQRNN
from visulize_results import *
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import model_from_json
from typhon.retrieval import qrnn


# load test data
set_nmb = 1
newXData, newYData = load_data(set_nmb)

quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
# input_dim = newXData.shape[1]


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

indexes = random.sample(range(0, len(xTrain)), len(xTrain))
x_val = xTrain[indexes[:20000]]
y_val = yTrain[indexes[:20000]]
x_train = xTrain[indexes[20000:]]
y_train = yTrain[indexes[20000:]]

input_dim = newXData.shape[1]

#load model
# model = extendedQRNN.QRNN((28*28*2+4,), quantiles, depth=8, width=256,
#                           activation = 'relu', model_name = 'CNN')
# model.fit(x_train = x_train,
#           y_train = y_train[:,3,3],
#           x_val = x_val,
#           y_val = y_val[:,3,3],
#           # batch_size = 512,
#           maximum_epochs = 1)

model = qrnn.QRNN.load('CNN_model.h5')

# xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
# yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
# times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
# distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')

prediction = model.predict(xTest)
pred = model.predict(x_val)

print(type(yTest[0,3,3]))
print(type(prediction[0,0]))
print(type(tf.Session().run(quantile_loss(np.float32(yTest[:,3,3]), prediction,
                                          quantiles))))

loss = tf.Session().run(quantile_loss(np.float32(yTest[:,3,3]), prediction,
                                      quantiles))

sum_rain = np.sum(yTest, axis = (1,2))
indexes = np.where(sum_rain > 400)[0]
index = indexes[10]
for i in range(5):
    plt.imshow(pred[index,:,:,i])
    plt.show()
plt.imshow(yTest[index,:,:])
print(np.mean(xTrain, axis=0, keepdims=True))
y_pred = model.predict(xTest)
crps = model.crps(y_pred, yTest[:,3,3], np.array(quantiles))
print(crps)
