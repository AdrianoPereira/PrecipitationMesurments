import numpy as np
import matplotlib.pyplot as plt

with open('./training_loss.txt', 'r') as file:
    losses = file.readlines()


def preprocess(line):
    data = line.replace('\n', '').replace(' ', '').split('-')[2:]
    data = [float(d.split(':')[1]) for d in data]
    return data


data = list(filter(lambda x: len(x) > 0, [preprocess(line)
                                          for line in losses]))
data = np.asarray(data)

plt.figure(figsize=(10, 7))
plt.title('Loss curve')
plt.plot(data[:, 0], color='blue', label='loss')
plt.plot(data[:, 1], color='green', label='val_loss')
plt.legend()
plt.savefig('loss_curve_mlp.png')
plt.show()
