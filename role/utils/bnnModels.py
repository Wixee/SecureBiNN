from os import stat
import numpy as np
from numpy.core.fromnumeric import choose

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, BatchNormalization, Activation, Flatten, MaxPool2D

from utils.bnnLayers import *
from utils.bnnActivations import *
from utils.coding import *
from copy import deepcopy
from utils.coding import extractbit, extractbit_tf


def get_bnn_mlp(input_shape, layers, bnn_scale, n_class, drop_rate=0.0):
    ret = Sequential()
    ret.add(Input(shape=input_shape))

    for i in range(len(layers)):
        n_units = layers[i]
        if i == 0:
            ret.add(Dense(units=n_units))
        else:
            ret.add(BNN_Dense(units=n_units))

        ret.add(BatchNormalization())
        ret.add(Activation(binarize))
        ret.add(DropoutNoScale(drop_rate))

    ret.add(Dense(n_class))
    ret.add(Activation('softmax'))
    return ret


def get_bnn_bm3(input_shape, layers, bnn_scale, n_class, drop_rate=0.0):

    ret = Sequential()
    ret.add(Input(shape=input_shape))
    ret.add(Conv2D(16*bnn_scale, kernel_size=(5, 5),
                   strides=(1, 1), data_format='channels_last'))
    ret.add(BatchNormalization())
    ret.add(Activation(binarize))
    ret.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))
    ret.add(DropoutNoScale(drop_rate))

    ret.add(BNN_Conv2D(16*bnn_scale, kernel_size=(5, 5),
                       strides=(1, 1), data_format='channels_last'))
    ret.add(BatchNormalization())
    ret.add(Activation(binarize))
    ret.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))
    ret.add(DropoutNoScale(drop_rate))

    ret.add(Flatten())
    for n_units in layers:
        ret.add(BNN_Dense(units=n_units*bnn_scale))
        ret.add(BatchNormalization())
        ret.add(Activation(binarize))
        ret.add(DropoutNoScale(drop_rate))

    ret.add(Dense(n_class))
    ret.add(Activation('softmax'))
    return ret


def load_bnn_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={
        'BNN_Dense': BNN_Dense,
        'Clip': Clip,
        'binarize': binarize,
        'binary_tanh': binarize,
        'DropoutNoScale': DropoutNoScale,
        'BNN_Conv2D': BNN_Conv2D
    })


def extract_layers_from_model(model):
    list_layers = list()

    for i in range(len(model.layers)):
        cur_layer = dict()

        name = model.layers[i].name
        cur_layer['name'] = name
        weights = model.layers[i].get_weights()

        # append the weights
        if name.startswith('dropout'):
            continue
        elif name.startswith('activation'):
            pass
        elif name.startswith('max_pooling'):
            cur_layer.update({
                'pool_size': model.get_layer(name).get_config()['pool_size'],
                'padding': model.get_layer(name).get_config()['padding'],
                'strides': model.get_layer(name).get_config()['strides']
            })
        elif name.startswith('batch_normalization'):
            [gamma, beta, mu, var] = weights
            deno = np.sqrt(var + 0.001)
            cur_layer['gamma'] = gamma / deno
            cur_layer['beta'] = beta - mu * gamma / deno
        elif name.startswith('dense'):
            cur_layer['W'] = weights[0]
            cur_layer['b'] = weights[1]
        elif name.startswith('conv2d'):
            cur_layer.update({
                'W': weights[0],
                'b': weights[1],
                'strides': model.get_layer(name).get_config()['strides'],
                'padding': model.get_layer(name).get_config()['padding']
            })
        elif name.startswith('bnn__dense'):
            weights = binary_tanh(weights[0]).numpy()
            cur_layer['W'] = weights
        elif name.startswith('bnn__conv2d'):
            cur_layer.update({
                'W': binary_tanh(np.array(weights)).numpy()[0],
                'strides': model.get_layer(name).get_config()['strides'],
                'padding': model.get_layer(name).get_config()['padding']
            })

        elif name.startswith('flatten'):
            pass

        list_layers.append(cur_layer)

    if list_layers[-1]['name'].startswith('activation'):
        del list_layers[-1]

    return list_layers


def inference(X, list_layers, record=False):

    ret = tf.Variable(X)
    if record:
        state = list()
    for layer in list_layers:
        name = layer['name']

        if name.startswith('dense'):
            ret = tf.matmul(ret, layer['W']) + layer['b']

        elif name.startswith('conv2d'):
            ret = tf.nn.conv2d(
                ret, layer['W'], strides=layer['strides'], padding=layer['padding'].upper())
            ret = tf.nn.bias_add(ret, layer['b'])

        elif name.startswith('batch_normalization'):
            ret = ret + layer['beta'] / layer['gamma']

        elif name.startswith('max_pooling'):
            ret = tf.nn.max_pool2d(
                ret, ksize=layer['pool_size'], strides=layer['strides'], padding=layer['padding'].upper())

        elif name.startswith('activation'):
            ret = binarize_tf(ret)

        elif name.startswith('bnn__dense'):
            ret = tf.matmul(ret, layer['W'])

        elif name.startswith('bnn__conv2d'):
            ret = tf.nn.conv2d(
                ret, layer['W'], strides=layer['strides'], padding=layer['padding'].upper())

        elif name.startswith('flatten'):
            ret = tf.reshape(ret, (ret.shape[0], -1))

        if record:
            state.append(ret)
    if record:
        return ret, state
    else:
        return ret


def choose_nbit_msb_from_shape(shape, name, fix=4):
    def choose_n_bit(x):
        if x < 2**6:
            return 8
        elif x < 2**15:
            return 16
        else:
            return 32

    input_nodes = None

    if name.find('conv') > -1:
        input_nodes = shape[:3]
    elif name.find('dense') > -1:
        input_nodes = shape[0]
    elif name.find('pool') > -1:
        input_nodes = shape

    # print(input_nodes)
    input_nodes = np.prod(input_nodes)

    msb = int(np.ceil(np.log2(input_nodes + 1))) + 1 - 1
    return choose_n_bit(input_nodes << fix), msb + fix


def code_list_layers(list_layers: list, precision=13, float_coded_dtype='int32', state=False):

    ret = list()

    def get_last_precision(i):
        return precision if i <= 0 else ret[i-1].get('precision', 0)

    def get_last_dtype(i):
        if i <= 0:
            return float_coded_dtype
        else:
            return ret[-1].get('nxt_dtype', ret[-1]['dtype'])

    for i in range(len(list_layers)):
        name = list_layers[i]['name']
        # print(i, name)
        cur_layer = deepcopy(list_layers[i])

        if name.startswith('dense') or name.startswith('conv2d'):

            n_bit = int(float_coded_dtype[3:])
            msb = n_bit - 1
            mod = 1 << msb

            last_precision = get_last_precision(i)
            W_coded = float2udtype(
                cur_layer['W'], float_coded_dtype, precision=precision)
            b_coded = float2udtype(
                cur_layer['b'], float_coded_dtype, precision=precision+last_precision)

            cur_layer.update({
                # basic property
                'dtype': float_coded_dtype,
                'mod': mod,
                'msb': msb,
                # others
                'W': W_coded,
                'b': b_coded,
                'precision': precision+last_precision
            })

        elif name.startswith('batch_normalization'):
            last_dtype = get_last_dtype(i)
            last_precision = get_last_precision(i)
            mod = ret[-1]['mod']

            # calculate val
            gamma, beta = cur_layer['gamma'], cur_layer['beta']
            del cur_layer['gamma'], cur_layer['beta']
            val = beta / gamma
            # clip the val
            last_n_bit = int(last_dtype[3:]) - last_precision - 1
            val = np.clip(val, -(2**(last_n_bit-1)), 2**(last_n_bit-1)-1)
            # val = np.ceil(val * 2 ** last_precision)
            val = -np.ceil(-val * 2 ** last_precision)
            # code the val
            val = float2udtype(
                val, float_coded_dtype=last_dtype, precision=0)

            cur_layer.update({
                # basic property
                'dtype': last_dtype,
                'mod': mod,
                'msb': ret[-1].get('nxt_msb', ret[-1]['msb']),
                # others
                'val': val
            })

        elif name.startswith('bnn__dense') or name.startswith('bnn__conv2d'):
            n_bit, msb = choose_nbit_msb_from_shape(cur_layer['W'].shape, name)
            dtype = 'int{}'.format(n_bit)
            mod = 1 << msb
            W_coded = float2udtype(cur_layer['W'], dtype, precision=0)
            cur_layer.update({
                # basic property
                'dtype': dtype,
                'mod': mod,
                'msb': msb,
                # others
                'W': W_coded
            })

        elif name.startswith('activation') or name.startswith('max_pooling'):
            if name.startswith('max_pooling'):
                assert(cur_layer['strides'] == cur_layer['pool_size'])

            last_dtype = get_last_dtype(i)

            nxt_pos = i + 1
            while(list_layers[nxt_pos]['name'].startswith('flatten')):
                nxt_pos = nxt_pos + 1
            nxt_name = list_layers[nxt_pos]['name']

            deact_val = None
            nxt_n_bit = None

            if nxt_name.startswith('max_pooling'):
                deact_val = 0
                nxt_n_bit, nxt_msb = choose_nbit_msb_from_shape(
                    list_layers[nxt_pos]['pool_size'],
                    nxt_name,
                    fix=0
                )
            elif nxt_name.startswith('bnn'):
                deact_val = -1
                nxt_n_bit, nxt_msb = choose_nbit_msb_from_shape(
                    list_layers[nxt_pos]['W'].shape,
                    nxt_name
                )
            elif nxt_name.startswith('dense'):
                deact_val = -1
                nxt_n_bit = int(float_coded_dtype[3:])
                nxt_msb = nxt_n_bit - 1

            nxt_mod = 1 << nxt_msb
            nxt_dtype = 'int{}'.format(nxt_n_bit)

            cur_layer.update({
                # basic property
                'dtype': last_dtype,
                'mod': ret[-1]['mod'],
                'msb': ret[-1].get('nxt_msb', ret[-1]['msb']),
                # others
                'deact_val': deact_val,
                'nxt_dtype': nxt_dtype,
                'nxt_msb': nxt_msb,
                'nxt_mod': nxt_mod
            })
        elif name.startswith('flatten'):
            cur_layer.update({
                # basic property
                'dtype': ret[-1]['dtype'],
                'mod': ret[-1]['mod'],
                'msb': ret[-1].get('nxt_msb', ret[-1]['msb'])
            })
        # print(cur_layer.keys())
        # print(cur_layer['dtype'], cur_layer['msb'])
        ret.append(cur_layer)

    return ret


def custom_model_summary(list_layers: list):
    for i in range(len(list_layers)):
        print('{}: {}'.format(i, list_layers[i]['name']), end=' ')
        if 'nxt_msb' in list_layers[i].keys():
            print('coding with {} bits.'.format(list_layers[i]['nxt_msb']+1))
        elif 'msb' in list_layers[i].keys():
            print('coding with {} bits.'.format(list_layers[i]['msb']+1))


def inference_coded_model(X_coded, list_layers: list, record=False):

    ret = tf.constant(X_coded.astype('int32'))
    if record:
        state = list()

    for i in range(len(list_layers)):
        name = list_layers[i]['name']
        cur_layer = list_layers[i]

        if name.startswith('dense'):
            ret = tf.matmul(ret, cur_layer['W'].astype(
                'int32')) + cur_layer['b'].astype('int32')

        elif name.startswith('conv2d'):
            ret = tf.nn.conv2d(ret, cur_layer['W'].astype(
                'int32'), strides=cur_layer['strides'], padding=cur_layer['padding'].upper())
            ret = tf.nn.bias_add(ret, cur_layer['b'].astype('int32'))

        elif name.startswith('batch_normalization'):
            ret = ret + cur_layer['val'].astype('int32')

        elif name.startswith('max_pooling') or name.startswith('activation'):
            if name.startswith('max_pooling'):
                ret = sum_pool(ret, kernel_size=cur_layer['pool_size'])
                ret = ret - 1

            # pos = int(cur_layer['dtype'][3:])-1
            pos = cur_layer['msb']
            ret = True ^ extractbit(ret.numpy(), pos, 32)
            if cur_layer['deact_val'] == -1:
                ret = 2 * ret - 1
            ret = float2udtype(ret, cur_layer['nxt_dtype'], precision=0)
            # ret = ret % cur_layer['nxt_mod']
            ret = tf.constant(ret, dtype=tf.int32)

        elif name.startswith('bnn__dense'):
            ret = tf.matmul(ret, cur_layer['W'].astype('int32'))

        elif name.startswith('bnn__conv2d'):
            ret = tf.nn.conv2d(ret, cur_layer['W'].astype(
                'int32'), strides=cur_layer['strides'], padding=cur_layer['padding'].upper())

        elif name.startswith('flatten'):
            ret = tf.reshape(ret, (ret.shape[0], -1))

        if record:
            state.append(ret)
    if record:
        return ret, state
    else:
        return ret


def code_model_to_shares(list_layer_coded):

    ret_0 = list()
    ret_1 = list()
    ret_2 = list()

    def assign_val(key, val_lst, n_hold):
        if n_hold == 2:
            ret_0[-1][key] = [val_lst[0], val_lst[1]]
            ret_1[-1][key] = [val_lst[1], val_lst[2]]
            ret_2[-1][key] = [val_lst[2], val_lst[0]]
        elif n_hold == 1:
            ret_0[-1][key] = val_lst[0]
            ret_1[-1][key] = val_lst[1]
            ret_2[-1][key] = val_lst[2]

    def split_val(key, ori_val, n_hold):
        assign_val(key, split_into_n_picies(ori_val, 3), n_hold)

    for i in range(len(list_layer_coded)):
        cur_layer = list_layer_coded[i]
        name = cur_layer['name']

        ret_0.append(deepcopy(cur_layer))
        ret_1.append(deepcopy(cur_layer))
        ret_2.append(deepcopy(cur_layer))

        if name.startswith('conv2d'):
            split_val('W', cur_layer['W'], 2)
            split_val('b', cur_layer['b'], 1)

        elif name.startswith('dense'):
            split_val('W', cur_layer['W'], 2)
            split_val('b', cur_layer['b'], 1)

        elif name.startswith('batch_normalization'):
            split_val('val', cur_layer['val'], 1)

        elif name.startswith('bnn__dense'):
            split_val('W', cur_layer['W'], 2)

        elif name.startswith('bnn__conv2d'):
            split_val('W', cur_layer['W'], 2)

    return [ret_0, ret_1, ret_2]
