import numpy as np


def extract_from_stream(stream: np.ndarray, shape_list: list):
    p = 0
    ret = list()

    for shape in shape_list:
        nxt_p = p + np.prod(shape)
        ret.append(stream[p:nxt_p].reshape(shape))
        p = nxt_p

    return ret


def choose(choose_bit: np.ndarray, x: np.ndarray, y: np.ndarray):
    return ((True ^ choose_bit) * x + choose_bit * y).astype(x.dtype)