'''
this code is originally from the typhoon package https://github.com/atmtools/typhon/blob/master/typhon/retrieval/qrnn/qrnn.py 
and developed by Simon Pfreundschuh. It was later altered by Gustav Tellwe to use CNN arcitecture instead of MLP. Not all of the
orininal documentation is thus valid and the biggest change is the addition of a model name to specify the CNN structure.
'''
import copy
import logging
import os
import pickle

import numpy as np
from scipy.interpolate import CubicSpline
# from Models import unet,convLSTM

# Keras Imports
try:
    import keras
    
    from keras.models import Model
    from keras.models import Sequential, clone_model
    from keras.layers import Dense, Activation, Conv2D,Conv3D, MaxPooling2D,MaxPooling3D, Flatten,BatchNormalization,Dropout, Input,concatenate
    from keras.optimizers import SGD, Adam,Adadelta,Adagrad,RMSprop
except ImportError:
    raise ImportError(
        "Could not import the required Keras modules. The QRNN "
        "implementation was developed for use with Keras version 2.0.9.")

################################################################################
# Loss Functions
################################################################################

import keras.backend as K


logger = logging.getLogger(__name__)


def skewed_absolute_error(y_true, y_pred, tau):
    """
    The quantile loss function for a given quantile tau:
    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)
    Where I is the indicator function.
    """
    #print(y_true)
    dy = y_pred - y_true
    return K.mean((1.0 - tau) * K.relu(dy) + tau * K.relu(-dy), axis=-1)


def quantile_loss(y_true, y_pred, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    #print(y_true.shape)
    #print(y_true)
    #print(y_pred)
    #e=0
    #print(K.int_shape(y_pred))
    #y_true = K.flatten(y_true)
    
    #y_pred = K.reshape(y_pred,(*K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]))
    e = skewed_absolute_error(K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
            e += skewed_absolute_error(K.flatten(y_true),
                                       K.flatten(y_pred[:, i + 1]),
                                       tau)
    '''
    for i in range(7):
        for j in range(7):
            
            tmp_y_pred = y_pred[:,i,j,:]
            tmp_y_true = y_true[:,i,j]
            
            e += skewed_absolute_error(
                K.flatten(tmp_y_true), K.flatten(tmp_y_pred[:, 0]), taus[0])
            for i, tau in enumerate(taus[1:]):
                e += skewed_absolute_error(K.flatten(tmp_y_true),
                                           K.flatten(tmp_y_pred[:, i + 1]),
                                           tau)
    '''
    return e


class QuantileLoss:
    """
    Wrapper class for the quantile error loss function. A class is used here
    to allow the implementation of a custom `__repr` function, so that the
    loss function object can be easily loaded using `keras.model.load`.
    Attributes:
        quantiles: List of quantiles that should be estimated with
                   this loss function.
    """

    def __init__(self, quantiles):
        self.__name__ = "QuantileLoss"
        self.quantiles = quantiles

    def __call__(self, y_true, y_pred):
        print(y_true.shape)
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"

################################################################################
# Keras Interface Classes
################################################################################

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_data, x_mean, x_sigma, y_data, sigma_noise, batch_size, shuffle):
        'Initialization'
        self.batch_size = batch_size
        self.x_data = x_data
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        self.y_data = y_data
        self.sigma_noise = sigma_noise
        self.shuffle = shuffle
        self.indexes = np.random.permutation(self.x_data.shape[0])
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print(int(np.floor(len(self.x_train) / self.batch_size)))
        return int(np.floor(len(self.x_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return (X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.indexes = np.random.permutation(self.x_data.shape[0])

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_data_batch = np.copy(self.x_data[indexes, :])
        x_data_batch = (x_data_batch - self.x_mean) / self.x_sigma
    
        
        y_batch = self.y_data[indexes]

        return (x_data_batch, y_batch)
class TrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.
    Attributes:
        x_train: The training input, i.e. the brightness temperatures
                 measured by the satellite.
        y_train: The training output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
        batch_size: The size of a training batch.
    """

    def __init__(self, x_train, x_mean, x_sigma, y_train, sigma_noise, batch_size,shuffle = True):
        self.bs = batch_size
        self.shuffle = shuffle
        self.x_train = x_train
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        self.y_train = y_train
        self.sigma_noise = sigma_noise

        self.indices = np.random.permutation(x_train.shape[0])
        self.i = 0

    def __iter__(self):
        logger.info("iter...")
        return self

    def __next__(self):
        inds = self.indices[np.arange(self.i * self.bs,
                                      (self.i + 1) * self.bs)
                            % self.indices.size]
        x_batch = np.copy(self.x_train[inds, :])
        if not self.sigma_noise is None:
            x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise
        x_batch = (x_batch - self.x_mean) / self.x_sigma
        y_batch = self.y_train[inds]

        self.i = self.i + 1

        # Shuffle training set after each epoch.
        if self.i % (self.x_train.shape[0] // self.bs) == 0:
            if self.shuffle:
                self.indices = np.random.permutation(self.x_train.shape[0])

        return (x_batch, y_batch)

# TODO: Make y-noise argument optional


class AdversarialTrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.
    Attributes:
        x_train: The training input, i.e. the brightness temperatures
                 measured by the satellite.
        y_train: The training output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
        batch_size: The size of a training batch.
    """

    def __init__(self,
                 x_train,
                 x_mean,
                 x_sigma,
                 y_train,
                 sigma_noise,
                 batch_size,
                 input_gradients,
                 eps):
        self.bs = batch_size

        self.x_train = x_train
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        self.y_train = y_train
        self.sigma_noise = sigma_noise

        self.indices = np.random.permutation(x_train.shape[0])
        self.i = 0

        # compile gradient function
        bs2 = self.bs // 2

        self.input_gradients = input_gradients
        self.eps = eps

    def __iter__(self):
        logger.info("iter...")
        return self

    def __next__(self):
        
        if self.i == 0:
            inds = np.random.randint(0, self.x_train.shape[0], self.bs)

            x_batch = np.copy(self.x_train[inds, :])
            if (self.sigma_noise):
                x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise

            x_batch = (x_batch - self.x_mean) / self.x_sigma
            y_batch = self.y_train[inds]

        else:

            bs2 = self.bs // 2
            inds = np.random.randint(0, self.x_train.shape[0], bs2)

            x_batch = np.zeros((self.bs, self.x_train.shape[1]))
            y_batch = np.zeros((self.bs, 1))

            x_batch[:bs2, :] = np.copy(self.x_train[inds, :])
            if (self.sigma_noise):
                x_batch[:bs2, :] += np.random.randn(bs2, self.x_train.shape[1]) \
                    * self.sigma_noise
            x_batch[:bs2, :] = (x_batch[:bs2, :] - self.x_mean) / self.x_sigma
            y_batch[:bs2, :] = self.y_train[inds].reshape(-1, 1)
            x_batch[bs2:, :] = x_batch[:bs2, :]
            y_batch[bs2:, :] = y_batch[:bs2, :]

            if (self.i > 10):
                grads = self.input_gradients(
                    [x_batch[:bs2, :], y_batch[:bs2, :], [1.0]])[0]
                x_batch[bs2:, :] += self.eps * np.sign(grads)

       
        self.i = self.i + 1
        
        
        return (x_batch, y_batch)


# TODO: Make y-noise argument optional
class ValidationGenerator:
    """
    This Keras sample generator is similar to the training generator
    only that it returns the whole validation set and doesn't perform
    any randomization.
    Attributes:
        x_val: The validation input, i.e. the brightness temperatures
                 measured by the satellite.
        y_val: The validation output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
    """

    def __init__(self, x_val, x_mean, x_sigma, y_val, sigma_noise):
        self.x_val = x_val
        self.x_mean = x_mean
        self.x_sigma = x_sigma

        self.y_val = y_val

        self.sigma_noise = sigma_noise

    def __iter__(self):
        return self

    def __next__(self):
        x_val = np.copy(self.x_val)
        if not self.sigma_noise is None:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        x_val = (x_val - self.x_mean) / self.x_sigma
        return (x_val, self.y_val)


class LRDecay(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.
    Attributes:
        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.
    """

    def __init__(self, model, lr_decay, lr_minimum, convergence_steps):
        self.model = model
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.convergence_steps = convergence_steps
        self.steps = 0

    def on_train_begin(self, logs={}):
        self.losses = []
        self.steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        self.losses += [logs.get('val_loss')]
        if not self.losses[-1] < self.min_loss:
            self.steps = self.steps + 1
        else:
            self.steps = 0
        if self.steps > self.convergence_steps:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(
                self.model.optimizer.lr, lr / self.lr_decay)
            self.steps = 0
            logger.info("\n Reduced learning rate to " + str(lr))

            if lr < self.lr_minimum:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])

################################################################################
# QRNN
################################################################################


class QRNN:
    r"""
    Quantile Regression Neural Network (QRNN)
    This class implements quantile regression neural networks and can be used
    to estimate quantiles of the posterior distribution of remote sensing
    retrievals.
    Internally, the QRNN uses a feed-forward neural network that is trained
    to minimize the quantile loss function
    .. math::
            \mathcal{L}_\tau(y_\tau, y_{true}) =
            \begin{cases} (1 - \tau)|y_\tau - y_{true}| & \text{ if } y_\tau < y_\text{true} \\
            \tau |y_\tau - y_\text{true}| & \text{ otherwise, }\end{cases}
    where :math:`x_\text{true}` is the expected value of the retrieval quantity
    and and :math:`x_\tau` is the predicted quantile. The neural network
    has one output neuron for each quantile to estimate.
    For the training, this implementation provides custom data generators that
    can be used to add Gaussian noise to the training data as well as adversarial
    training using the fast gradient sign method.
    This implementation also provides functionality to use an ensemble of networks
    instead of just a single network to predict the quantiles.
    .. note:: For the QRNN I am using :math:`x` to denote the input vector and
              :math:`y` to denote the output. While this is opposed to typical
              inverse problem notation, it is inline with machine learning
              notation and felt more natural for the implementation. If this
              annoys you, I am sorry. But the other way round it would have
              annoyed other people and in particular me.
    Attributes:
        input_dim (int):
            The input dimension of the neural network, i.e. the dimension of the
            measurement vector.
        quantiles (numpy.array):
            The 1D-array containing the quantiles :math:`\tau \in [0, 1]` that the
            network learns to predict.
        depth (int):
            The number layers in the network excluding the input layer.
        width (int):
            The width of the hidden layers in the network.
        activation (str):
            The name of the activation functions to use in the hidden layers
            of the network.
        models (list of keras.models.Sequential):
            The ensemble of Keras neural networks used for the quantile regression
            neural network.
    """

    def __init__(self,
                 input_dim,
                 quantiles,
                 depth=3,
                 width=128,
                 activation="relu",
                 ensemble_size=1,
                 model_name = None,
                 **kwargs):
        """
        Create a QRNN model.
        Arguments:
            input_dim(int): The dimension of the measurement space, i.e. the number
                            of elements in a single measurement vector y
            quantiles(np.array): 1D-array containing the quantiles  to estimate of
                                 the posterior distribution. Given as fractions
                                 within the range [0, 1].
            depth(int): The number of hidden layers  in the neural network to
                        use for the regression. Default is 3, i.e. three hidden
                        plus input and output layer.
            width(int): The number of neurons in each hidden layer.
            activation(str): The name of the activation functions to use. Default
                             is "relu", for rectified linear unit. See 
                             `this <https://keras.io/activations>`_ link for
                             available functions.
            model_name: This is a sting that determines what type of newtork will be used
            **kwargs: Additional keyword arguments are passed to the constructor
                      call `keras.layers.Dense` of the hidden layers, which can
                      for example be used to add regularization. For more info consult
                      `Keras documentation. <https://keras.io/layers/core/#dense>`_
            
        """
        self.input_dim = input_dim
        self.quantiles = np.array(quantiles)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.model_name = model_name

        model = Sequential()
        if model_name == 'convLSTM':
            model = convLSTM()
        elif model_name == 'unet':
            model = unet(input_size = (112,112,2))
        elif model_name == 'CNN':
            momentum = 0.9
            epsilon = 0.001
            activation = 'relu'
            start_kernels = 32
            drop = 0.3
            input1 = Input( shape = input_dim)
            input2 = Input( shape = (4,) )
            #conv_model = Sequential()
            tmp_input = input1
        
            tmp_input = Conv2D(64, kernel_size=(3,3),
                             input_shape = input_dim,
                             padding = 'same',
                             activation = activation)(tmp_input)
            tmp_input = Conv2D(64, kernel_size=(3,3),
                             padding = 'same',
                             activation = activation)(tmp_input)
            
            tmp_input = MaxPooling2D(pool_size=(2, 2))(tmp_input)
            
            tmp_input = Conv2D(128, kernel_size=(3, 3),
                             padding = 'same',
                             activation = activation)(tmp_input)
            tmp_input = Conv2D(128, kernel_size=(3, 3),
                             padding = 'same',
                             activation = activation)(tmp_input)
            
            tmp_input = MaxPooling2D(pool_size=(2,2))(tmp_input)
            
            tmp_input = Conv2D(256, kernel_size=(3,3),
                             padding = 'same',
                             activation = activation)(tmp_input)
            tmp_input = Conv2D(256, kernel_size=(3, 3),
                             padding = 'same',
                             activation = activation)(tmp_input)
            
            tmp_input = MaxPooling2D(pool_size=(2, 2))(tmp_input)
            
            
            flat = Flatten()(tmp_input)
            
            #combine = concatenate(cnn_output, input2)
            #print(cnn_output)
            #print(input2)
            #concat = concatenate([flat, input2])
            #fc_model.add(Flatten())
            
            #model.add(Dropout(0.25))
            dense = Dense(256, activation='relu')(flat)
            dense = Dense(256, activation='relu')(dense)
            dense = Dense(256, activation='relu')(dense)
            out = Dense(5)(dense)
            
            model = Model(inputs = input1, outputs = out)
            print(model.summary())
            

            
            
            #output = fc_model( [ cnn_output , input2 ] )

            #model = Model( [ input1 , input2 ] , output )
           
            
            
            print('model')
        elif model_name == 'MLP':
            
            model = Sequential()
            
            model.add(Dense(width, activation = activation,input_shape=input_dim))
            for i in range(depth -2):
                model.add(Dense(width, activation = activation))
                
            model.add(Dense(len(quantiles)))
            
       
        
        print('add')
        self.models = [clone_model(model) for i in range(ensemble_size)]

    def __fit_params__(self, kwargs):
        at = kwargs.pop("adversarial_training", False)
        dat = kwargs.pop("delta_at", 0.01)
        batch_size = kwargs.pop("batch_size", 512)
        convergence_epochs = kwargs.pop("convergence_epochs", 10)
        initial_learning_rate = kwargs.pop('initial_learning_rate', 0.01)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 2.0)
        learning_rate_minimum = kwargs.pop('learning_rate_minimum', 1e-6)
        maximum_epochs = kwargs.pop("maximum_epochs", 200)
        training_split = kwargs.pop("training_split", 0.9)
        return at, dat, batch_size, convergence_epochs, initial_learning_rate, \
            learning_rate_decay, learning_rate_minimum, maximum_epochs, \
            training_split, kwargs

    def cross_validation(self,
                        x_train,
                        y_train,
                        sigma_noise = None,
                        n_folds=5,
                        s=None,
                        **kwargs):
        r"""
        Perform n-fold cross validation.
        This function trains the network n times on different subsets of the
        provided training data, always keeping a fraction of 1/n samples apart
        for testing. Performance for each of the networks is evaluated and mean
        and standard deviation for all folds are returned. This is to reduce
        the influence of random fluctuations of the network performance during
        hyperparameter tuning.
        Arguments:
            x_train(numpy.array): Array of shape :code:`(m, n)` containing the
                                  m n-dimensional training inputs.
            y_train(numpy.array): Array of shape :code:`(m, 1)` containing the
                                  m training outputs.
            sigma_noise(None, float, np.array): If not `None` this value is used
                                                to multiply the Gaussian noise
                                                that is added to each training
                                                batch. If None no noise is
                                                added.
            n_folds(int): Number of folds to perform for the cross correlation.
            s(callable, None): Performance metric for the fold. If not None,
                               this should be a function object taking as
                               arguments :code:`(y_test, y_pred)`, i.e. the
                               expected output for the given fold :code:`y_test`
                               and the predicted output :code:`y_pred`. The
                               returned value is taken as performance metric.
            **kwargs: Additional keyword arguments are passed on to the :code:`fit`
                      method that is called for each fold.
        """

        n = x_train.shape[0]
        n_test = n // n_folds
        inds = np.random.permutation(np.arange(0, n))

        results = []

        # Nomenclature is a bit difficult here:
        # Each cross validation fold has its own training,
        # vaildation and test set. The size of the test set
        # is number of provided training samples divided by the
        # number of fold. The rest is used a traning and internal
        # validation set according to the chose training_split
        # ratio.


        for i in range(n_folds):
            for m in self.models:
                m.reset_states()

            # Indices to use for training including training data and internal
            # validation set to monitor convergence.
            inds_train = np.append(np.arange(0, i * n_test),
                                       np.arange(min((i + 1) * n_test, n), n))
            inds_train = inds[inds_train]
            # Indices used to evaluate performance of the model.
            inds_test = np.arange(i * n_test, (i + 1) * n_test)
            inds_test = inds[inds_test]

            x_test_fold = x_train[inds_test, :]
            y_test_fold = y_train[inds_test]

            # Splitting training and validation set.
            x_train_fold = x_train[inds_train, :]
            y_train_fold = y_train[inds_train]

            self.fit(x_train_fold, y_train_fold,
                     sigma_noise, **kwargs)

            # Evaluation on this folds test set.
            if s:
                y_pred = self.predict(x_test_fold)
                results += [s(y_pred, y_test_fold)]
            else:
                loss = self.models[0].evaluate(
                    (x_test_fold - self.x_mean) / self.x_sigma,
                    y_test_fold)
                logger.info(loss)
                results += [loss]
        logger.info(results)
        results = np.array(results)
        logger.info(results)
        return (np.mean(results, axis=0), np.std(results, axis=0))

    def fit(self,
            x_train,
            y_train,
            x_val,
            y_val,
            sigma_noise=None,
            **kwargs):
        r"""
        Train the QRNN on given training data.
        The training uses an internal validation set to monitor training
        progress. This can be either split at random from the training
        data (see `training_fraction` argument) or provided explicitly
        using the `x_val` and `y_val` parameters
        Training of the QRNN is performed using stochastic gradient descent
        (SGD). The learning rate is adaptively reduced during training when
        the loss on the internal validation set has not been reduced for a
        certain number of epochs.
        Two data augmentation techniques can be used to improve the
        calibration of the QRNNs predictions. The first one adds Gaussian
        noise to training batches before they are passed to the network.
        The noise statistics are defined using the `sigma_noise` argument.
        The second one is adversarial training. If adversarial training is
        used, half of each training batch consists of adversarial samples
        generated using the fast gradient sign method. The strength of the
        perturbation is controlled using the `delta_at` parameter.
        During one epoch, each training sample is passed once to the network.
        Their order is randomzied between epochs.
        Arguments:
            x_train(np.array): Array of shape `(n, m)` containing n training
                               samples of dimension m.
            y_train(np.array): Array of shape `(n, )` containing the training
                               output corresponding to the training data in
                               `x_train`.
            sigma_noise(None, float, np.array): If not `None` this value is used
                                                to multiply the Gaussian noise
                                                that is added to each training
                                                batch. If None no noise is
                                                added.
            x_val(np.array): Array of shape :code:`(n', m)` containing n' validation
                             inputs that will be used to monitor training loss. Must
                             be provided in unison with :code:`y_val` or otherwise
                             will be ignored.
            y_val(np.array): Array of shape :code:`(n')` containing n'  validation
                             outputs corresponding to the inputs in :code:`x_val`.
                             Must be provided in unison with :code:`x_val` or
                             otherwise will be ignored.
            adversarial_training(Bool): Whether or not to use adversarial training.
                                        `False` by default.
            delta_at(flaot): Perturbation factor for the fast gradient sign method
                             determining the strength of the adversarial training
                             perturbation. `0.01` by default.
            batch_size(float): The batch size to use during training. Defaults to `512`.
            convergence_epochs(int): The number of epochs without decrease in
                                     validation loss before the learning rate
                                     is reduced. Defaults to `10`.
            initial_learning_rate(float): The inital value for the learning
                                          rate.
            learning_rate_decay(float): The factor by which to reduce the
                                        learning rate after no improvement
                                        on the internal validation set was
                                        observed for `convergence_epochs`
                                        epochs. Defaults to `2.0`.
            learning_rate_minimum(float): The minimum learning rate at which
                                          the training is terminated. Defaults
                                          to `1e-6`.
            maximum_epochs(int): The maximum number of epochs to perform if
                                 training does not terminate before.
            training_split(float): The ratio `0 < ts < 1.0` of the samples in
                                   to be used as internal validation set. Defaults
                                   to 0.9.
        """
        #if not (x_train[0,:].shape == self.input_dim):
        #    raise Exception("Training input must have the same extent along dimension 1 as input_dim (" + str(self.input_dim)+ ")")

        #if not (y_train.shape[1] == 1):
        #    raise Exception("Currently only scalar retrieval targets are supported.")

        x_mean = np.mean(x_train, axis=0, keepdims=True)
        x_sigma = np.std(x_train, axis=0, keepdims=True)
        self.x_mean = x_mean
        self.x_sigma = x_sigma

        # Handle parameters
        # at:  adversarial training
        # bs:  batch size
        # ce:  convergence epochs
        # ilr: initial learning rate
        # lrd: learning rate decay
        # lrm: learning rate minimum
        # me:  maximum number of epochs
        # ts:  split ratio of training set
        at, dat, bs, ce, ilr, lrd, lrm, me, ts, kwargs = self.__fit_params__(
            kwargs)

        # Split training and validation set if x_val or y_val
        # are not provided.
        #n = x_train.shape[0]
        #n_train = n
       
        loss = QuantileLoss(self.quantiles)

        self.custom_objects = {loss.__name__: loss}
        for model in self.models:
            
            optimizer = SGD(lr=ilr)
            #optimizer = Adagrad()
            #optimizer = Adam()
            #optimizer = RMSprop(learning_rate=0.001)
            
            model.compile(loss=loss, optimizer=optimizer)
      
            training_generator = DataGenerator(x_train, self.x_mean, self.x_sigma,
                                                       y_train, sigma_noise, bs, True)
            validation_generator = DataGenerator(x_val, self.x_mean, self.x_sigma,
                                                       y_val, sigma_noise,bs,False)
            lr_callback = LRDecay(model, lrd, lrm, ce)
            '''
            model.fit_generator(training_generator, steps_per_epoch=n_train // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1, callbacks=[lr_callback])
            
            '''
            model.fit_generator(training_generator, 
                                epochs=me, validation_data=validation_generator,
                                callbacks=[lr_callback])
            
            
    def predict(self, x):
        r"""
        Predict quantiles of the conditional distribution P(y|x).
        Forward propagates the inputs in `x` through the network to
        obtain the predicted quantiles `y`.
        Arguments:
            x(np.array): Array of shape `(n, m)` containing `n` m-dimensional inputs
                         for which to predict the conditional quantiles.
        Returns:
             Array of shape `(n, k)` with the columns corresponding to the k
             quantiles of the network.
        """
        predictions = np.stack(
            [m.predict((x - self.x_mean) / self.x_sigma) for m in self.models])
        return np.mean(predictions, axis=0)

    def cdf(self, x):
        r"""
        Approximate the posterior CDF for given inputs `x`.
        Propagates the inputs in `x` forward through the network and
        approximates the posterior CDF by a piecewise linear function.
        The piecewise linear function is given by its values at
        approximate quantiles $x_\tau$ for
        :math: `\tau = \{0.0, \tau_1, \ldots, \tau_k, 1.0\}` where
        :math: `\tau_k` are the quantiles to be estimated by the network.
        The values for :math:`x_0.0` and :math:`x_1.0` are computed using
        .. math::
            x_0.0 = 2.0 x_{\tau_1} - x_{\tau_2}
            x_1.0 = 2.0 x_{\tau_k} - x_{\tau_{k-1}}
        Arguments:
            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.
        Returns:
            Tuple (xs, fs) containing the :math: `x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        y_pred = np.zeros(self.quantiles.size + 2)
        y_pred[1:-1] = self.predict(x)
        y_pred[0] = 2.0 * y_pred[1] - y_pred[2]
        y_pred[-1] = 2.0 * y_pred[-2] - y_pred[-3]

        qs = np.zeros(self.quantiles.size + 2)
        qs[1:-1] = self.quantiles
        qs[0] = 0.0
        qs[-1] = 1.0

        return y_pred, qs

    def pdf(self, x, use_splines = False):
        r"""
        Approximate the posterior probability density function (PDF) for given
        inputs `x`.
        By default, the PDF is approximated by computing the derivative of the
        piece-wise linear approximation of the CDF as computed by the :code:`cdf`
        function.
        If :code:`use_splines` is set to :code:`True`, the PDF is computed from
        a spline fit to the approximate CDF.
        Arguments:
            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.
            use_splines(bool): Whether or not to use a spline fit to the CDF to
            approximate the PDF.
        Returns:
            Tuple (xs, fs) containing the :math: `x`-values in `xs` and corresponding
            values of the approximate posterior PDF :math: `F(x)` in `fs`.
        """

        y_pred = np.zeros(self.quantiles.size)
        y_pred = self.predict(x).ravel()

        y = np.zeros(y_pred.size + 1)
        y[1:-1] = 0.5 * (y_pred[1:] + y_pred[:-1])
        y[0] = 2 * y_pred[0] - y_pred[1]
        y[-1] = 2 * y_pred[-1] - y_pred[-2]

        if not use_splines:

            p = np.zeros(y.size)
            p[1:-1] = np.diff(self.quantiles) / np.diff(y_pred)
        else:

            y = np.zeros(y_pred.size + 2)
            y[1:-1] = y_pred
            y[0] = 3 * y_pred[0] - 2 * y_pred[1]
            y[-1] = 3 * y_pred[-1] - 2 * y_pred[-2]
            q = np.zeros(self.quantiles.size + 2)
            q[1:-1] = np.array(self.quantiles)
            q[0] = 0.0
            q[-1] = 1.0

            sr = CubicSpline(y, q, bc_type = "clamped")
            y = np.linspace(y[0], y[-1], 101)
            p = sr(y, nu = 1)

        return y, p


        y_pred = np.zeros(self.quantiles.size + 2)
        y_pred[1:-1] = self.predict(x)
        y_pred[0] = 2.0 * y_pred[1] - y_pred[2]
        y_pred[-1] = 2.0 * y_pred[-2] - y_pred[-3]

        if use_splines:
            x_t = np.zeros(x.size + 2)
            x_t[1:-1] = x
            x_t[0] = 2 * x[0] - x[1]
            x_t[-1] = 2 * x[-1] - x[-2]
            y_t = np.zeros(y.size + 2)
            y_t[1:-1] = y
            y_t[-1] = 1.0

        else:
            logger.info(y)
            x_new = np.zeros(x.size - 1)
            x_new[2:-2] = 0.5 * (x[2:-3] + x[3:-2])
            x_new[0:2] = x[0:2]
            x_new[-2:] = x[-2:]
            y_new = np.zeros(y.size - 1)
            y_new[1:-1] = np.diff(y[1:-1]) / np.diff(x[1:-1])
        return x_new, y_new

    def sample_posterior(self, x, n=1):
        r"""
        Generates :code:`n` samples from the estimated posterior
        distribution for the input vector :code:`x`. The sampling
        is performed by the inverse CDF method using the estimated
        CDF obtained from the :code:`cdf` member function.
        Arguments:
            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.
            n(int): The number of samples to generate.
        Returns:
            Tuple (xs, fs) containing the :math: `x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        y_pred, qs = self.cdf(x)
        p = np.random.rand(n)
        y = np.interp(p, qs, y_pred)
        return y

    def posterior_mean(self, x):
        r"""
        Computes the posterior mean by computing the first moment of the
        estimated posterior CDF.
        Arguments:
            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the posterior mean.
        Returns:
            Array containing the posterior means for the provided inputs.
        """
        y_pred, qs = self.cdf(x)
        mus = y_pred[-1] - np.trapz(qs, x=y_pred)
        return mus

    @staticmethod
    def crps(y_pred, y_test, quantiles):
        r"""
        Compute the Continuous Ranked Probability Score (CRPS) for given quantile
        predictions.
        This function uses a piece-wise linear fit to the approximate posterior
        CDF obtained from the predicted quantiles in :code:`y_pred` to
        approximate the continuous ranked probability score (CRPS):
        .. math::
            CRPS(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
            - \mathrm{1}_{x < x'})^2 \: dx'
        Arguments:
            y_pred(numpy.array): Array of shape `(n, k)` containing the `k`
                                 estimated quantiles for each of the `n`
                                 predictions.
            y_test(numpy.array): Array containing the `n` true values, i.e.
                                 samples of the true conditional distribution
                                 estimated by the QRNN.
            quantiles: 1D array containing the `k` quantile fractions :math:`\tau`
                       that correspond to the columns in `y_pred`.
        Returns:
            `n`-element array containing the CRPS values for each of the
            predictions in `y_pred`.
        """
        y_cdf = np.zeros((y_pred.shape[0], quantiles.size + 2))
        y_cdf[:, 1:-1] = y_pred
        y_cdf[:, 0] = 2.0 * y_pred[:, 1] - y_pred[:, 2]
        y_cdf[:, -1] = 2.0 * y_pred[:, -2] - y_pred[:, -3]

        ind = np.zeros(y_cdf.shape)
        ind[y_cdf > y_test.reshape(-1, 1)] = 1.0

        qs = np.zeros((1, quantiles.size + 2))
        qs[0, 1:-1] = quantiles
        qs[0, 0] = 0.0
        qs[0, -1] = 1.0
        print(y_cdf.shape)
        print(qs.shape)
        return np.trapz((qs - ind)**2.0, y_cdf)

    def evaluate_crps(self, x, y_test):
        r"""
        Predict quantiles and compute the Continuous Ranked Probability Score (CRPS).
        This function evaluates the networks prediction on the
        inputs in `x` and evaluates the CRPS of the predictions
        against the materializations in `y_test`.
        Arguments:
            x(numpy.array): Array of shape `(n, m)` containing the `n`
                            `m`-dimensional inputs for which to evaluate
                            the CRPS.
            y_test(numpy.array): Array containing the `n` materializations of
                                 the true conditional distribution.
        Returns:
            `n`-element array containing the CRPS values for each of the
            inputs in `x`.
        """
        return QRNN.crps(self.predict(x), y_test, self.quantiles)

    def save(self, path):
        r"""
        Store the QRNN model in a file.
        This stores the model to a file using pickle for all
        attributes that support pickling. The Keras model
        is handled separately, since it can not be pickled.
        .. note:: In addition to the model file with the given filename,
                  additional files suffixed with :code:`_model_i` will be
                  created for each neural network this model consists of.
        Arguments:
            path(str): The path including filename indicating where to
                       store the model.
        """

        f = open(path, "wb")
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dirname = os.path.dirname(path)

        self.model_files = []
        for i, m in enumerate(self.models):
            self.model_files += [name + "_model_" + str(i)]
            m.save(os.path.join(dirname, self.model_files[i]))
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(path):
        r"""
        Load a model from a file.
        This loads a model that has been stored using the `save` method.
        Arguments:
            path(str): The path from which to read the model.
        Return:
            The loaded QRNN object.
        """
        filename = os.path.basename(path)
        dirname = os.path.dirname(path)

        f = open(path, "rb")
        qrnn = pickle.load(f)
        qrnn.models = []
        for mf in qrnn.model_files:
            mf = os.path.basename(mf)
            try:
                mp = os.path.join(dirname, os.path.basename(mf))
                qrnn.models += [keras.models.load_model(mp, qrnn.custom_objects)]
            except:
                raise Exception("Error loading the neural network models. " \
                                "Please make sure all files created during the"\
                                " saving are in this folder.")
        f.close()
        return qrnn

    def __getstate__(self):
        dct = copy.copy(self.__dict__)
        dct.pop("models")
        return dct

    def __setstate__(self, state):
        self.__dict__ = state
        self.models = None


