import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Dense, Conv2D
from utils.bnnActivations import binarize, binary_tanh, binary_sigmoid


class Clip(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        super(Clip, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {
            "min_value": self.min_value,
            "max_value": self.max_value
        }


class BNN_Conv2D(Conv2D):
    def __init__(self, filters, scale=1, kernel_lr_multiplier='Glorot', **kwargs):
        super(BNN_Conv2D, self).__init__(filters, **kwargs)
        self.filters = filters
        self.scale = scale
        self.kernel_lr_multiplier = kernel_lr_multiplier

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel_constraint = Clip(-self.scale, self.scale)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

    def call(self, inputs):
        binary_kernel = binary_tanh(self.kernel)
        outputs = K.conv2d(
            inputs,
            binary_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class BNN_Dense(Dense):

    def __init__(self, units, clip_bound=1.0, **kwargs):
        self.units = units
        self.clip_bound = clip_bound
        super(BNN_Dense, self).__init__(self.units, **kwargs)

    def build(self, input_shape):
        self.kernel_constraint = Clip(-self.clip_bound, self.clip_bound)
        self.kernel = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), self.units])

    def call(self, inputs):
        w_binary = binary_tanh(self.kernel)
        return K.dot(inputs, w_binary)


class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
    '''

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs


def sum_pool(x, kernel_size):
    ret = tf.reshape(x, (x.shape[0], x.shape[1]//kernel_size[0], kernel_size[0],
                         x.shape[2]//kernel_size[1], kernel_size[1], x.shape[3]))
    ret = tf.reduce_sum(ret, axis=(2, 4))
    return ret
