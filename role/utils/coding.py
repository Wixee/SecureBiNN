import numpy as np
import tensorflow as tf
from copy import deepcopy


def int2uint(x: np.ndarray):
    '''
    transfer int_bits to uint_bits
    [-ring_size / 2, ring_size/2-1) -> [0, ring_size)
    '''
    return x.astype('u'+str(x.dtype))


def uint2int(x: np.ndarray):
    '''
    transfer uint_bits to int_bits
    [0, ring_size) -> [-ring_size / 2, ring_size/2-1)
    '''
    bits = int(str(x.dtype)[4:])
    signed_dtype = 'int{}'.format(bits)
    ring_size = 2**bits
    threshold = ring_size >> 1
    idx = x >= threshold

    ret = deepcopy(x)
    ret = x.astype('int64')
    ret[idx] -= ring_size
    return ret.astype(signed_dtype)


def code(x: np.ndarray, coded_dtype='uint32'):
    '''
    transfer int_bits to uint_bits
    [-ring_size / 2, ring_size/2-1) -> [0, ring_size) 
    '''
    return x.astype(coded_dtype)


def decode(x: np.ndarray, bits: int, plain_dtype):
    '''
    transfer uint_bits to int_bits
    [0, ring_size) -> [-ring_size / 2, ring_size/2-1]
    '''
    ring_size = 2**bits
    threshold = ring_size >> 1
    idx = x >= threshold

    ret = x.astype('int64')
    ret[idx] -= ring_size
    return ret.astype(plain_dtype)


# def split_into_n_picies(x_coded: np.ndarray, n_piece: int, bits: int, dtype: str):
#     ring_size = 2**bits
#     shape = x_coded.shape
#     ret = np.random.randint(low=0, high=ring_size,
#                             size=(n_piece, *shape)).astype(dtype)
#     ret[-1] = x_coded - np.sum(ret[:-1], axis=0)
#     return ret


def split_into_n_picies(x_coded: np.ndarray, n_piece: int):
    dtype = str(x_coded.dtype)
    assert dtype.startswith(
        'uint'), 'The dtype of the input is {}.'.format(dtype)
    n_bit = int(dtype[4:])
    ring_size = 2 ** n_bit
    ret = np.random.randint(low=0, high=ring_size, size=(
        n_piece, *x_coded.shape), dtype=dtype)
    ret[-1] = x_coded - np.sum(ret[:-1], axis=0)
    return ret


def float2int(x: np.ndarray, precision: int, dtype):
    return (x * 2**precision).astype(dtype)


def int2float(x: np.ndarray, precision: int, dtype='float32'):
    return x.astype(dtype) / (2**precision)


def float2udtype(x: np.ndarray, float_coded_dtype: str, precision: int):
    return code(float2int(x, precision, float_coded_dtype), 'u'+float_coded_dtype)


def udtype2float(x: np.ndarray, float_coded_dtype: str, precision: int):
    n_bit = int(float_coded_dtype[3:])
    return int2float(decode(x, n_bit, plain_dtype=float_coded_dtype), precision)

# def float2udtype(x: np.ndarray, float_coded_dtype: str, precision: int):
#     return int2uint(float2int(x, precision, float_coded_dtype))


# def udtype2float(x: np.ndarray, float_coded_dtype: str, precision: int):
#     return int2float(uint2int(x), precision)

def extractbit(x: np.ndarray, pos: int, n_bits: int):
    return np.right_shift(np.left_shift(x, n_bits-pos-1), n_bits-1).astype('bool')


def extractbit_tf(x: tf.Tensor, pos: int, n_bits: int):
    return tf.bitwise.right_shift(tf.bitwise.left_shift(x, n_bits-pos-1), n_bits-1)
