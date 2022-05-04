import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

_epsilon = 0.0000001


def round_through(x):
    '''
    Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    '''
    Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830)
    '''
    x = 0.5 * x + 0.5 + _epsilon
    return K.clip(x, 0.0, 1.0)


def binary_sigmoid(x):
    '''
    Binary hard sigmoid for training binarized neural network.

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830)
    '''
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    '''
    Binary hard sigmoid for training binarized neural network.
     The neurons' activations binarization function
     It behaves like the sign function during forward propagation
     And like:
        hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
        clear gradient when |x| > 1 during back propagation

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830)
    '''
    return 2 * binary_sigmoid(x) - 1


def binarize(W):
    '''
    Binarize the inputs.
    '''
    return binary_tanh(W)


def binarize_np(X):
    return 2 * np.round(np.clip(0.5*X+0.5+_epsilon, 0.0, 1.0)) - 1


def binarize_tf(X):
    return 2 * tf.round(tf.clip_by_value(0.5*X+0.5+_epsilon, 0.0, 1.0)) - 1
