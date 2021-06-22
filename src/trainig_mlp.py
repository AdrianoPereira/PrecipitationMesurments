from data_loader_preprocess import preprocess_data
import extendedQRNN
from sklearn.model_selection import train_test_split
import numpy as np
import os

# load data
PATH = "/home/adriano/Desktop/PrecipitationMesurments/data/MLP"
OUTPUT = "/home/adriano/Desktop/PrecipitationMesurments/data/output"
x_train = np.load(f"{PATH}/MLP_x_val.npy")
y_train = np.load(f"{PATH}/MLP_y_val.npy")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=.9)
print(f"Train shapes: {x_train.shape}, {y_train.shape}")
print(f"Validation shapes: {x_val.shape}, {y_val.shape}")

# define configuration model
model_name = 'MLP'
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
model = extendedQRNN.QRNN(
    (28*28*2+4,), quantiles, depth=8, width=256, activation='relu',
    model_name='MLP'
)

# train model
batch_size = 32
epochs = 20
model.fit(
    x_train=x_train,
    y_train=y_train[:, 3, 3],
    x_val=x_val,
    y_val=y_val[:, 3, 3],
    batch_size=batch_size,
    maximum_epochs=500
)

model.save(f'{OUTPUT}/mlp_model_new_data_trained.h5')

