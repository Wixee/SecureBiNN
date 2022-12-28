import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import dtype, print_tensor
from utils.thread import custom_Thread
from utils.randomTool import RandomHandler
from utils.npTcp import npTcpSender, npTcpReceiver
from utils.randomTool import get_random_mask, RandomHandler
from utils.coding import *
from utils.bnnLayers import sum_pool
from utils.np_ops import choose, extract_from_stream


def calculate_2_input_gate(a_masked: np.ndarray, b_masked: np.ndarray,
                           triple: np.ndarray, constant_computation: bool):

    ret = triple[2] ^ (a_masked & triple[1]) ^ (b_masked & triple[0])

    if constant_computation:
        ret ^= a_masked & b_masked

    return ret


def beaver_mult_bool(x_share: np.ndarray, y_share: np.ndarray, triple,
                     sender: npTcpSender, listener: npTcpReceiver,
                     constant_computation: bool):
    '''
    return z_share = x_share * y_share
    '''
    x_share_masked = x_share ^ triple[0]
    y_share_masked = y_share ^ triple[1]
    msg = np.array([x_share_masked, y_share_masked], dtype='bool')

    send_msg_thread = sender.send_ndarray_by_bytes_thread(msg, packbits=True)
    send_msg_thread.start()

    [x_other, y_other] = listener.receive_ndarray_by_bytes(size=msg.shape,
                                                           packbits=True)

    x_masked = x_share_masked ^ x_other
    y_masked = y_share_masked ^ y_other

    z_share = calculate_2_input_gate(x_masked, y_masked, triple,
                                     constant_computation)

    send_msg_thread.join()

    return z_share


def beaver_mult_bool_two_round_with_optimization(x_share: np.ndarray,
                                                 y_share: np.ndarray, triple,
                                                 sender: npTcpSender,
                                                 listener: npTcpReceiver,
                                                 constant_computation: bool):
    '''
    return z_share = x_share * y_share
    '''
    # modify d = x ^ u
    d_share = x_share ^ triple[0]
    e_share = y_share ^ triple[1]

    # print('d_share.shape:', d_share.shape)
    # print('e_share.shape:', e_share.shape)
    # print('x_share.shape:', x_share.shape)
    # print('y_share.shape:', y_share.shape)
    # print('triple[0].shape', triple[0].shape)
    # print('triple[1].shape:', triple[1].shape)
    # print(d.shape)
    # assert(np.sum(d_share[0]!=d_share[1])==0)
    # only send one element of d

    msg = np.array([d_share[0], e_share[0], e_share[1]], dtype='bool')

    send_msg_thread = sender.send_ndarray_by_bytes_thread(msg, packbits=True)
    send_msg_thread.start()
    rcv_thread = listener.receive_ndarray_by_bytes_thread(size=msg.shape,
                                                          packbits=True)
    rcv_thread.start()
    rcv_thread.join()

    [d_other_0, e_other_0, e_other_1] = rcv_thread.get_result()

    # recover the array
    d_other = np.array([d_other_0, d_other_0], dtype='bool')
    e_other = np.array([e_other_0, e_other_1], dtype='bool')

    d = d_share ^ d_other
    e = e_share ^ e_other
    z_share = triple[2] ^ (e & triple[0]) ^ (d & triple[1])

    if constant_computation:
        z_share ^= d & e

    send_msg_thread.join()

    return z_share


def calculate_3_input_gate(a_masked: np.ndarray, b_masked: np.ndarray,
                           c_masked: np.ndarray, help_tuple,
                           constant_computation: bool):
    # help_tuple: (u, v, w, uv, uw, vw, uvw)

    ret = (help_tuple[3] & c_masked) ^ (help_tuple[4] & b_masked) ^ (
        help_tuple[5] & a_masked) ^ (a_masked & b_masked & help_tuple[2]) ^ (
            a_masked & help_tuple[1] & c_masked) ^ (help_tuple[0] & b_masked
                                                    & c_masked) ^ help_tuple[6]

    if constant_computation:
        ret ^= a_masked & b_masked & c_masked

    return ret


def beaver_mult_bool_3_input(a_share: np.ndarray, b_share: np.ndarray,
                             c_share: np.ndarray, help_tuple,
                             sender: npTcpSender, listener: npTcpReceiver,
                             constant_computation: bool):
    '''
    return z_share = a_share * b_share * c_share
    help_tuple: (u, v, w, uv, uw, vw, uvw)
    '''
    a_share_masked = a_share ^ help_tuple[0]
    b_share_masked = b_share ^ help_tuple[1]
    c_share_masked = c_share ^ help_tuple[2]

    msg = np.array([a_share_masked, b_share_masked, c_share_masked],
                   dtype='bool')

    send_msg_thread = sender.send_ndarray_by_bytes_thread(msg, packbits=True)
    send_msg_thread.start()
    rcv_thread = listener.receive_ndarray_by_bytes_thread(size=msg.shape,
                                                          packbits=True)
    rcv_thread.start()

    rcv_thread.join()
    [a_other, b_other, c_other] = rcv_thread.get_result()

    a_masked = a_share_masked ^ a_other
    b_masked = b_share_masked ^ b_other
    c_masked = c_share_masked ^ c_other

    z_share = calculate_3_input_gate(a_masked, b_masked, c_masked, help_tuple,
                                     constant_computation)

    send_msg_thread.join()

    return z_share


def ot_3party(role,
              sender: list,
              listener: list,
              random_handler: RandomHandler,
              msg_0=None,
              msg_1=None,
              msg_dtype=None,
              msg_bit=None,
              msg_shape=None,
              choose_bit=None):
    '''
    role=0 means receiver
    role=1 means sender
    role=2 means role_ttp
    '''

    role_receiver = 0
    role_sender = 1
    role_ttp = 2

    ret = None

    if role == role_receiver:

        ori_shape = choose_bit.shape

        # get same random numbers with sender
        swap_bit = random_handler.get_same_random_with(role_sender,
                                                       bits=1,
                                                       size=ori_shape)
        random_handler.cnter += 1

        # Send choose_bit
        sender[role_ttp].send_ndarray_by_bytes(choose_bit ^ swap_bit,
                                               packbits=True)

        # receive the message
        ret = listener[role_ttp].receive_ndarray_by_bytes(size=ori_shape,
                                                          dtype=msg_dtype)

        # unmask the message
        mask = random_handler.get_same_random_with(role_sender,
                                                   bits=msg_bit,
                                                   size=(2, *ori_shape))
        random_handler.cnter += 1
        mask = ((True ^ choose_bit ^ swap_bit) * mask[0] +
                (choose_bit ^ swap_bit) * mask[1]).astype(ret.dtype)

        ret = ret ^ mask

    elif role == role_sender:

        # get same random numbers with receiver and swap the message
        swap_bit = random_handler.get_same_random_with(role_receiver,
                                                       bits=1,
                                                       size=msg_0.shape)
        random_handler.cnter += 1
        msg = np.zeros(shape=(2, *msg_0.shape), dtype=msg_0.dtype)
        msg[0] = choose(swap_bit, msg_0, msg_1)
        msg[1] = choose(True ^ swap_bit, msg_0, msg_1)

        # mask the message
        mask = random_handler.get_same_random_with(role_receiver,
                                                   bits=msg_bit,
                                                   size=(2, *msg_0.shape))
        random_handler.cnter += 1
        msg = msg ^ mask
        # send
        sender[role_ttp].send_ndarray_by_bytes(msg)

    elif role == role_ttp:

        # receive the choose bit from receiver
        choice_thread = listener[
            role_receiver].receive_ndarray_by_bytes_thread(msg_shape,
                                                           packbits=True)
        choice_thread.start()

        # receive the masked message from sender
        data_thread = listener[role_sender].receive_ndarray_by_bytes_thread(
            (2, *msg_shape), dtype=msg_dtype)
        data_thread.start()

        # keep the counter in sync
        random_handler.cnter += 1

        choice_thread.join()
        choose_bit = choice_thread.get_result()
        data_thread.join()
        data = data_thread.get_result()

        # select the message
        msg = ((True ^ choose_bit) * data[0] + choose_bit * data[1]).astype(
            data.dtype)

        # send the selected message to receiver
        sender[role_receiver].send_ndarray_by_bytes(msg)

    return ret


def extract_msb(role: int, listener: list, sender: list,
                random_handler: RandomHandler, a_share: np.ndarray,
                b_share: np.ndarray, total_bit: int, pos: int,
                triple: np.ndarray):
    data_owner = 0
    model_owner = 1
    ret = None
    if role == model_owner or role == data_owner:
        const_compute = True if role == model_owner else False
        p_triple = 0
        a_1 = extractbit(a_share, pos=0, n_bits=total_bit)
        b_1 = extractbit(b_share, pos=0, n_bits=total_bit)

        # 2)
        c_last = beaver_mult_bool(a_1, b_1, triple[:,
                                                   p_triple], sender[1 - role],
                                  listener[1 - role], const_compute)
        p_triple += 1

        for i in range(1, pos + 1):
            a_i = extractbit(a_share, i, total_bit)
            b_i = extractbit(b_share, i, total_bit)
            y_i = a_i ^ b_i
            # d)
            if i == pos:
                ret = y_i ^ c_last
            else:
                # a)
                d_i = beaver_mult_bool(a_i, b_i, triple[:, p_triple],
                                       sender[1 - role], listener[1 - role],
                                       const_compute)
                if const_compute:
                    d_i ^= True
                p_triple += 1

                # b)
                e_i = beaver_mult_bool(y_i, c_last, triple[:, p_triple],
                                       sender[1 - role], listener[1 - role],
                                       const_compute)
                p_triple += 1
                if const_compute:
                    e_i ^= True

                # c)
                c_i = beaver_mult_bool(e_i, d_i, triple[:, p_triple],
                                       sender[1 - role], listener[1 - role],
                                       const_compute)
                p_triple += 1
                if const_compute:
                    c_i ^= True
                c_last = c_i

        ret ^= random_handler.get_same_random_with(1 - role,
                                                   bits=1,
                                                   size=ret.shape)

    # all three party add the cnter
    random_handler.cnter += 1

    return ret


def count_ppa_n_triple(msb_pos: int):
    # msb_pos counts from zero
    ret = msb_pos
    l_bound = msb_pos - 1
    while l_bound > 0:
        is_even = False

        if l_bound % 2 == 0:
            is_even = True
            l_bound = l_bound - 1

        n_node = (l_bound + 1) // 2
        ret += n_node + n_node

        if is_even == True:
            # count the unpaired node
            l_bound = n_node + 1 - 1
        else:
            l_bound = n_node - 1
    return ret


def modify_triple_with_optimization(msb_pos, triple_do=None, triple_mo=None):
    # msb_pos is counted from zero
    # there are msb_pos + 1 bits

    p_triple = 0

    # calculate carry_bit
    p_triple += msb_pos

    # divide and conquer
    l_bound = msb_pos - 1

    while l_bound > 0:
        exist_unpaired_bits = False

        if l_bound % 2 == 0:
            l_bound = l_bound - 1
            exist_unpaired_bits = True

        # l_bound is odd
        l_idx = list(range(1, l_bound + 1, 2))
        len_l = len(l_idx)
        # len_l = l_bound // 2 + 1

        # data_1 = np.array([l_all_1, l_all_1], dtype='bool')
        # data_2 = np.array([r_all_1, carry_bit[r_idx]], dtype='bool')

        # [U, U'] -> [U, U]
        mid_p = p_triple + len_l
        nxt_p = mid_p + len_l

        # print(triple_do.shape)
        # print(triple_mo.shape)

        if triple_do is not None:
            triple_do[0, mid_p:nxt_p] = triple_do[0, p_triple:mid_p]
        if triple_mo is not None:
            triple_mo[0, mid_p:nxt_p] = triple_mo[0, p_triple:mid_p]
        p_triple = nxt_p

        if exist_unpaired_bits is True:
            len_l += 1

        l_bound = len_l - 1
        # print(l_bound, len_l)

    if (triple_do is not None) and (triple_mo is not None):
        delta = (triple_do[0] ^ triple_mo[0]) & (triple_do[1]
                                                 ^ triple_mo[1]) ^ triple_do[2]
        return delta
    elif triple_mo is not None:
        return triple_mo
    elif triple_do is not None:
        return triple_do

    return None


def extract_msb_ppa(role: int, listener: list, sender: list,
                    a_share: np.ndarray, b_share: np.ndarray, total_bit: int,
                    msb_pos: int, triple: np.ndarray):

    # msb_pos and l_bound counts from zero
    if role == 2:
        return None

    const_compute = True if role == 0 else False

    a_share_bit = list()
    b_share_bit = list()

    ret = extractbit(a_share, msb_pos, n_bits=total_bit) ^ extractbit(
        b_share, msb_pos, n_bits=total_bit)

    for i in range(0, msb_pos):
        a_share_bit.append(extractbit(a_share, i, n_bits=total_bit))
        b_share_bit.append(extractbit(b_share, i, n_bits=total_bit))

    a_share_bit = np.array(a_share_bit, dtype='bool')
    b_share_bit = np.array(b_share_bit, dtype='bool')

    p_triple = 0

    all_1 = a_share_bit ^ b_share_bit

    nxt_p = p_triple + len(a_share_bit)
    carry_bit = beaver_mult_bool(a_share_bit,
                                 b_share_bit,
                                 triple[:, p_triple:nxt_p],
                                 sender[1 - role],
                                 listener[1 - role],
                                 constant_computation=const_compute)
    p_triple = nxt_p

    l_bound = msb_pos - 1

    while l_bound > 0:
        # All the nodes are paired

        left_most_all_1 = None
        left_most_carry_bit = None

        if l_bound % 2 == 0:
            left_most_all_1 = all_1[l_bound]
            left_most_carry_bit = carry_bit[l_bound]
            l_bound = l_bound - 1

        l_idx = list(range(1, l_bound + 1, 2))
        r_idx = list(range(0, l_bound + 1, 2))
        l_all_1 = all_1[l_idx]
        r_all_1 = all_1[r_idx]

        data_1 = np.array([l_all_1, l_all_1], dtype='bool')
        data_2 = np.array([r_all_1, carry_bit[r_idx]], dtype='bool')

        nxt_p = p_triple + len(l_all_1) + len(l_all_1)
        data_ret = beaver_mult_bool(data_1,
                                    data_2,
                                    triple[:, p_triple:nxt_p].reshape(
                                        3, *data_1.shape),
                                    sender[1 - role],
                                    listener[1 - role],
                                    constant_computation=const_compute)
        p_triple = nxt_p

        all_1 = data_ret[0]
        carry_bit = data_ret[1] ^ carry_bit[l_idx]

        if left_most_all_1 is not None:
            all_1 = np.append(all_1, [left_most_all_1], axis=0)
            carry_bit = np.append(carry_bit, [left_most_carry_bit], axis=0)

        l_bound = len(all_1) - 1

    # print(p_triple, triple.shape[1],
    #         msb_pos, count_ppa_n_triple(msb_pos))
    return ret ^ carry_bit[0]


def extract_msb_ppa_with_optimize(role: int, listener: list, sender: list,
                                  a_share: np.ndarray, b_share: np.ndarray,
                                  total_bit: int, msb_pos: int,
                                  triple: np.ndarray):

    # msb_pos and l_bound counts from zero
    if role == 2:
        return None

    const_compute = True if role == 0 else False

    a_share_bit = list()
    b_share_bit = list()

    ret = extractbit(a_share, msb_pos, n_bits=total_bit) ^ extractbit(
        b_share, msb_pos, n_bits=total_bit)

    for i in range(0, msb_pos):
        a_share_bit.append(extractbit(a_share, i, n_bits=total_bit))
        b_share_bit.append(extractbit(b_share, i, n_bits=total_bit))

    a_share_bit = np.array(a_share_bit, dtype='bool')
    b_share_bit = np.array(b_share_bit, dtype='bool')

    p_triple = 0

    all_1 = a_share_bit ^ b_share_bit

    nxt_p = p_triple + len(a_share_bit)
    carry_bit = beaver_mult_bool(a_share_bit,
                                 b_share_bit,
                                 triple[:, p_triple:nxt_p],
                                 sender[1 - role],
                                 listener[1 - role],
                                 constant_computation=const_compute)
    p_triple = nxt_p

    l_bound = msb_pos - 1

    while l_bound > 0:

        # make nodes be paired
        left_most_all_1 = None
        left_most_carry_bit = None

        if l_bound % 2 == 0:
            left_most_all_1 = all_1[l_bound]
            left_most_carry_bit = carry_bit[l_bound]
            l_bound = l_bound - 1

        l_idx = list(range(1, l_bound + 1, 2))
        r_idx = list(range(0, l_bound + 1, 2))
        l_all_1 = all_1[l_idx]
        r_all_1 = all_1[r_idx]

        data_1 = np.array([l_all_1, l_all_1], dtype='bool')
        data_2 = np.array([r_all_1, carry_bit[r_idx]], dtype='bool')

        mid_p = p_triple + len(l_all_1)
        nxt_p = mid_p + len(l_all_1)
        cur_triple = triple[:, p_triple:nxt_p]
        cur_triple[0, mid_p:nxt_p] = cur_triple[0, p_triple:mid_p]
        data_ret = beaver_mult_bool_two_round_with_optimization(
            data_1,
            data_2,
            cur_triple.reshape(3, *data_1.shape),
            sender[1 - role],
            listener[1 - role],
            constant_computation=const_compute)
        p_triple = nxt_p

        all_1 = data_ret[0]
        carry_bit = data_ret[1] ^ carry_bit[l_idx]

        if left_most_all_1 is not None:
            all_1 = np.append(all_1, [left_most_all_1], axis=0)
            carry_bit = np.append(carry_bit, [left_most_carry_bit], axis=0)

        l_bound = len(all_1) - 1

    return ret ^ carry_bit[0]


def extract_msb_with_3_input_gate(role: int, listener: list, sender: list,
                                  a_share: np.ndarray, b_share: np.ndarray,
                                  total_bit: int, msb_pos: int,
                                  triple: np.ndarray, help_tuple: np.ndarray):
    '''
    help_tuple: (u, v, w, uv, uw, vw, uvw)
    '''
    # msb_pos and l_bound counts from zero
    if role == 2:
        return None

    const_compute = True if role == 0 else False

    # extract MSB
    ret = extractbit(a_share, msb_pos, n_bits=total_bit) ^ extractbit(
        b_share, msb_pos, n_bits=total_bit)

    a_share_bit = list()
    b_share_bit = list()

    # extract other bits
    for i in range(0, msb_pos):
        a_share_bit.append(extractbit(a_share, i, n_bits=total_bit))
        b_share_bit.append(extractbit(b_share, i, n_bits=total_bit))
    a_share_bit = np.array(a_share_bit, dtype='bool')
    b_share_bit = np.array(b_share_bit, dtype='bool')

    p_triple = 0
    p_help_tuple = 0

    # Start to solve the problem of the leaves

    all_1 = a_share_bit ^ b_share_bit
    p_triple_nxt = p_triple + len(a_share_bit)
    carry_bit = beaver_mult_bool(a_share_bit,
                                 b_share_bit,
                                 triple[:, p_triple:p_triple_nxt],
                                 sender[1 - role],
                                 listener[1 - role],
                                 constant_computation=const_compute)
    p_triple = p_triple_nxt

    # divide and conquer
    l_bound = msb_pos - 1

    while l_bound > 0:

        flag = (l_bound + 1) % 3
        n_paired = (l_bound + 1) // 3
        paired_l_bound = max(n_paired * 3 - 1, 0)

        l_idx = list(range(2, paired_l_bound + 1, 3))
        m_idx = list(range(1, paired_l_bound + 1, 3))
        r_idx = list(range(0, paired_l_bound + 1, 3))

        # the input of 3-input-and gate
        # print(l_bound, all_1.shape, carry_bit.shape, n_paired)
        l_all_1 = all_1[l_idx]
        m_all_1 = all_1[m_idx]
        r_all_1 = all_1[r_idx]

        l_carry_bit = carry_bit[l_idx]
        m_carry_bit = carry_bit[m_idx]
        r_carry_bit = carry_bit[r_idx]

        # the input of 2-input-and gate (the left most two bits)
        left_most_l_all_1 = None
        left_most_l_carry_bit = None
        left_most_r_all_1 = None
        left_most_r_carry_bit = None

        left_most_all_1 = None
        left_most_carry_bit = None

        if flag == 1:
            left_most_all_1 = all_1[l_bound:l_bound + 1]
            left_most_carry_bit = carry_bit[l_bound:l_bound + 1]

        elif flag == 2:
            left_most_l_all_1 = all_1[l_bound:l_bound + 1]
            left_most_l_carry_bit = carry_bit[l_bound:l_bound + 1]

            left_most_r_all_1 = all_1[l_bound - 1:l_bound]
            left_most_r_carry_bit = carry_bit[l_bound - 1:l_bound]

        cur_node_len = n_paired  # the number of 3-input gates
        p_help_tuple_nxt = p_help_tuple + cur_node_len

        # mask the input
        if n_paired > 0:
            masked_tuple_elements_sent = np.array([
                l_all_1 ^ help_tuple[0, p_help_tuple:p_help_tuple_nxt],
                m_all_1 ^ help_tuple[1, p_help_tuple:p_help_tuple_nxt],
                r_all_1 ^ help_tuple[2, p_help_tuple:p_help_tuple_nxt],
                r_carry_bit ^ help_tuple[7, p_help_tuple:p_help_tuple_nxt],
                m_carry_bit ^ help_tuple[11, p_help_tuple:p_help_tuple_nxt]
            ])
            data_sent = masked_tuple_elements_sent.reshape(-1)

        if flag == 2:
            masked_triple_elements_sent = np.array([
                left_most_l_all_1 ^ triple[0, p_triple:p_triple + 1],
                left_most_r_all_1 ^ triple[1, p_triple:p_triple + 1],
                left_most_l_all_1 ^ triple[0, p_triple + 1:p_triple + 2],
                left_most_r_carry_bit ^ triple[1, p_triple + 1:p_triple + 2]
            ],
                dtype='bool')
            if n_paired == 0:
                data_sent = masked_triple_elements_sent.reshape(-1)
            else:
                # print('l_bound:', l_bound)
                # print('data_sent.shape:', data_sent.shape)
                # print('masked_triple_elements_sent.shape:', masked_triple_elements_sent.shape)
                data_sent = np.append(masked_tuple_elements_sent.reshape(-1),
                                      masked_triple_elements_sent.reshape(-1),
                                      axis=0)

        # exchange the message
        send_msg_thread = sender[1 - role].send_ndarray_by_bytes_thread(
            data_sent, packbits=True)
        send_msg_thread.start()
        data_received = listener[1 - role].receive_ndarray_by_bytes(
            size=data_sent.shape, packbits=True)

        # recover the masked data

        data_recovered = data_sent ^ data_received

        p_stream = 0
        if n_paired > 0:
            p_stream_nxt = p_stream + np.prod(masked_tuple_elements_sent.shape)
            masked_tuple_elements = data_recovered[
                p_stream:p_stream_nxt].reshape(
                    masked_tuple_elements_sent.shape)
            p_stream = p_stream_nxt
            l_all_1_masked = masked_tuple_elements[0]
            m_all_1_masked = masked_tuple_elements[1]
            r_all_1_masked = masked_tuple_elements[2]
            r_carry_bit_masked = masked_tuple_elements[3]
            m_carry_bit_masked = masked_tuple_elements[4]

        if flag == 2:
            p_stream_nxt = p_stream + np.prod(
                masked_triple_elements_sent.shape)
            masked_triple_elements = data_recovered[
                p_stream:p_stream_nxt].reshape(
                    masked_triple_elements_sent.shape)
            p_stream = p_stream_nxt
            left_most_l_all_1_masked_1 = masked_triple_elements[0]
            left_most_r_all_1_masked = masked_triple_elements[1]
            left_most_l_all_1_masked_2 = masked_triple_elements[2]
            left_most_r_carry_bit_masked = masked_triple_elements[3]

        if n_paired > 0:
            # all_1 = l_all_1 & m_all_1 & r_all_1
            all_1 = calculate_3_input_gate(
                l_all_1_masked, m_all_1_masked, r_all_1_masked,
                help_tuple[[0, 1, 2, 3, 4, 5, 6],
                           p_help_tuple:p_help_tuple_nxt], const_compute)

            # carry_bit = (l_all_1 & m_all_1 & r_carry_bit) ^ (l_all_1 & m_carry_bit) ^ l_carry_bit
            carry_bit = calculate_3_input_gate(
                l_all_1_masked, m_all_1_masked, r_carry_bit_masked,
                help_tuple[[0, 1, 7, 3, 8, 9, 10],
                           p_help_tuple:p_help_tuple_nxt],
                const_compute) ^ calculate_2_input_gate(
                    l_all_1_masked, m_carry_bit_masked,
                    help_tuple[[0, 11, 12], p_help_tuple:p_help_tuple_nxt],
                    const_compute) ^ l_carry_bit

            p_help_tuple = p_help_tuple_nxt

        if flag == 2:
            # left_most_all_1 = left_most_l_all_1 & left_most_r_all_1
            left_most_all_1 = calculate_2_input_gate(
                left_most_l_all_1_masked_1, left_most_r_all_1_masked,
                triple[:, p_triple:p_triple + 1], const_compute)

            # left_most_carry_bit = left_most_l_all_1 & left_most_r_carry_bit ^ left_most_l_carry_bt
            left_most_carry_bit = calculate_2_input_gate(
                left_most_l_all_1_masked_2, left_most_r_carry_bit_masked,
                triple[:, p_triple + 1:p_triple + 2],
                const_compute) ^ left_most_l_carry_bit

            p_triple = p_triple + 2

        if left_most_all_1 is not None:
            # all_1 = np.append(all_1, left_most_all_1, axis=0)
            if n_paired == 0:
                carry_bit = left_most_carry_bit
                all_1 = left_most_all_1
            else:
                carry_bit = np.append(carry_bit, left_most_carry_bit, axis=0)
                all_1 = np.append(all_1, left_most_all_1, axis=0)

        l_bound = len(all_1) - 1
        send_msg_thread.join()

    return ret ^ carry_bit[0]


def verify_triple(triple: np.ndarray, sender, listener):
    send_msg_thread = sender.send_ndarray_by_bytes_thread(triple,
                                                          packbits=True)
    send_msg_thread.start()
    rcv_thread = listener.receive_ndarray_by_bytes_thread(size=triple.shape,
                                                          packbits=True)
    rcv_thread.start()

    send_msg_thread.join()
    rcv_thread.join()

    triple_recieved = rcv_thread.get_result()

    tmp = triple ^ triple_recieved
    assert (np.sum((tmp[0] & tmp[1]) != tmp[2]) == 0)


def count_tuple_and_triple(msb_pos: int):
    '''
    return (n_tuple, n_triple)
    '''
    n_tuple, n_triple = 0, msb_pos

    l_bound = msb_pos - 1

    while l_bound > 0:
        flag = (l_bound + 1) % 3

        if flag == 2:
            n_triple += 2

        n_pair = (l_bound + 1) // 3
        n_tuple += n_pair
        l_bound = n_pair - 1

        if flag != 0:
            l_bound += 1

    return n_tuple, n_triple


def get_triple_delta(triple_mo: np.ndarray, triple_do: np.ndarray):
    '''
    get the delta sent to model owner
    '''
    return (triple_mo[0] ^ triple_do[0]) & (triple_mo[1]
                                            ^ triple_do[1]) ^ triple_do[2]


def get_tuple_delta(tuple_mo: np.ndarray, tuple_do: np.ndarray):
    '''
    get the delta sent to model owner
    tuple:
    u,  v,  w,  uv, uw, vw, uvw
    0,  1,  2,   3,  4,  5, 6

    u,  v,  w1, uv, uw1,    vw1,    uvw1
    0,  1,  7,  3,  8,      9,      10

    u,  v2, uv2
    0,  11, 12
    '''
    u = tuple_mo[0] ^ tuple_do[0]
    v = tuple_mo[1] ^ tuple_do[1]
    w = tuple_mo[2] ^ tuple_do[2]
    w1 = tuple_mo[7] ^ tuple_do[7]
    v2 = tuple_mo[11] ^ tuple_do[11]

    ret = np.array([
        u & v ^ tuple_do[3], u & w ^ tuple_do[4], v & w ^ tuple_do[5],
        u & v & w ^ tuple_do[6], u & w1 ^ tuple_do[8], v & w1 ^ tuple_do[9],
        u & v & w1 ^ tuple_do[10], u & v2 ^ tuple_do[12]
    ],
        dtype='bool')
    return ret


def SecureBiNN(role, random_handler, listener, sender, data_share,
               model_share):
    # role 0: data owner (share_0, share_1)
    # role 1: model owner (share_1, share_2)
    # role 2: trusted third party (share_2, share_0)
    data_owner = 0
    model_owner = 1
    ttp = 2

    role_last = (role - 1) % 3
    role_next = (role + 1) % 3

    ret = data_share

    for i in range(len(model_share)):

        random_handler.step_synchronous()

        cur_layer = model_share[i]
        name = cur_layer['name']
        # print(name)
        # print('Len ret:', len(ret))

        if name.startswith('dense'):
            ret = tf.matmul(ret[0], cur_layer['W'][0]) + \
                tf.matmul(ret[0], cur_layer['W'][1]) + \
                tf.matmul(ret[1], cur_layer['W'][0]) + \
                cur_layer['b']
            ret += random_handler.get_3_out_of_3_shares(bits=int(
                cur_layer['dtype'][3:]),
                size=ret.shape)
            random_handler.cnter += 1

        elif name.startswith('conv2d'):
            ret = tf.nn.conv2d(
                ret[0], cur_layer['W'][0], strides=cur_layer['strides'], padding=cur_layer['padding']) + \
                tf.nn.conv2d(
                    ret[0], cur_layer['W'][1], strides=cur_layer['strides'], padding=cur_layer['padding']) + \
                tf.nn.conv2d(
                    ret[1], cur_layer['W'][0], strides=cur_layer['strides'], padding=cur_layer['padding'])
            ret = tf.nn.bias_add(ret, cur_layer['b'])
            ret += random_handler.get_3_out_of_3_shares(bits=int(
                cur_layer['dtype'][3:]),
                size=ret.shape)
            random_handler.cnter += 1

        elif name.startswith('batch_normalization'):
            ret += cur_layer['val']

        elif name.startswith('activation') or name.startswith('max_pooling'):
            if name.startswith('max_pooling'):
                ret = ret[0]
                ret = sum_pool(ret, kernel_size=cur_layer['pool_size'])
                if role == data_owner:
                    ret = ret - 1

            input_dtype = cur_layer['dtype']
            output_dtype = cur_layer['nxt_dtype']

            total_bit = 32
            input_bit = int(input_dtype[3:])
            pos = cur_layer['msb']

            output_bit = int(output_dtype[3:])
            ori_shape = ret.shape
            ret = ret.numpy()

            # n_triple = 3 * (pos - 1) + 1
            # n_triple = count_ppa_n_triple(msb_pos=pos)
            n_tuple, n_triple = count_tuple_and_triple(msb_pos=pos)

            # Prepare beaver's triple
            '''
            tuple:
            u,  v,  w,  uv, uw, vw, uvw
            0,  1,  2,   3,  4,  5, 6

            u,  v,  w1, uv, uw1,    vw1,    uvw1
            0,  1,  7,  3,  8,      9,      10

            u,  v2, uv2
            0,  11, 12

            generate tuple, then generate triple
            '''

            if role == ttp:
                if name.startswith('activation'):
                    send_datashare_thread = sender[
                        data_owner].send_ndarray_by_bytes_thread(ret)
                    send_datashare_thread.start()

                tuple_do = random_handler.get_same_random_with(
                    data_owner, bits=1, size=(13, n_tuple, *ret.shape))
                tuple_mo = random_handler.get_same_random_with(
                    model_owner, bits=1, size=(13, n_tuple, *ret.shape))
                random_handler.cnter += 1

                triple_do = random_handler.get_same_random_with(
                    data_owner, bits=1, size=(3, n_triple, *ret.shape))
                triple_mo = random_handler.get_same_random_with(
                    model_owner, bits=1, size=(3, n_triple, *ret.shape))
                random_handler.cnter += 1

                # delta = modify_triple_with_optimization(
                #     pos, triple_do, triple_mo)
                # delta = (triple_do[0] ^ triple_mo[0]) & (
                #     triple_do[1] ^ triple_mo[1]) ^ triple_do[2]

                delta_tuple = get_tuple_delta(tuple_mo, tuple_do)
                delta_triple = get_triple_delta(triple_mo, triple_do)
                delta = np.append(delta_tuple.reshape(-1),
                                  delta_triple.reshape(-1))

                send_delta_thread = sender[
                    model_owner].send_ndarray_by_bytes_thread(delta,
                                                              packbits=True)
                send_delta_thread.start()
                a_share = None
                b_share = None
                triple = None
                tuple = None
                send_delta_thread.join()

                if name.startswith('activation'):
                    send_datashare_thread.join()

            elif role == data_owner:
                if name.startswith('activation'):
                    ttp_share = listener[ttp].receive_ndarray_by_bytes(
                        size=ret.shape, dtype=ret.dtype)
                    ret += ttp_share

                # triple = random_handler.get_same_random_with(
                #     ttp, bits=1, size=(3, n_triple, *ret.shape))
                # triple = modify_triple_with_optimization(
                #     pos, triple_do=triple)
                # random_handler.cnter += 1

                tuple = random_handler.get_same_random_with(ttp,
                                                            bits=1,
                                                            size=(13, n_tuple,
                                                                  *ret.shape))
                random_handler.cnter += 1

                triple = random_handler.get_same_random_with(ttp,
                                                             bits=1,
                                                             size=(3, n_triple,
                                                                   *ret.shape))
                random_handler.cnter += 1

                a_share = ret
                b_share = np.zeros_like(ret)

            elif role == model_owner:

                # delta = listener[ttp].receive_ndarray_by_bytes(
                #     size=(n_triple, *ret.shape), packbits=True)
                # triple = random_handler.get_same_random_with(
                #     ttp, bits=1, size=(3, n_triple, *ret.shape))
                # triple = modify_triple_with_optimization(
                #     pos, triple_mo=triple)
                # random_handler.cnter += 1
                # triple[2] = delta

                tuple = random_handler.get_same_random_with(ttp,
                                                            bits=1,
                                                            size=(13, n_tuple,
                                                                  *ret.shape))
                random_handler.cnter += 1

                triple = random_handler.get_same_random_with(ttp,
                                                             bits=1,
                                                             size=(3, n_triple,
                                                                   *ret.shape))
                random_handler.cnter += 1

                delta = listener[ttp].receive_ndarray_by_bytes(
                    size=((8 * n_tuple + n_triple) * np.prod(ret.shape)),
                    packbits=True)

                len_delta_triple = np.prod(triple[2].shape)
                len_delta_tuple = len(delta) - len_delta_triple

                delta_tuple = np.reshape(delta[:len_delta_tuple],
                                         (-1, n_tuple, *ret.shape))
                delta_triple = np.reshape(delta[len_delta_tuple:],
                                          triple[2].shape)

                tuple[3] = delta_tuple[0]
                tuple[4] = delta_tuple[1]
                tuple[5] = delta_tuple[2]
                tuple[6] = delta_tuple[3]
                tuple[8] = delta_tuple[4]
                tuple[9] = delta_tuple[5]
                tuple[10] = delta_tuple[6]
                tuple[12] = delta_tuple[7]

                triple[2] = delta_triple

                a_share = np.zeros_like(ret)
                b_share = ret

            # if role != ttp:
            #     verify_triple(triple, sender[1-role], listener[1-role])
            # Extract MSB
            # ret = extract_msb_ppa(
            #     role=role,
            #     listener=listener,
            #     sender=sender,
            #     a_share=a_share,
            #     b_share=b_share,
            #     total_bit=total_bit,
            #     msb_pos=pos,
            #     triple=triple
            # )
            # ret = extract_msb_ppa_with_optimize(
            #     role=role,
            #     listener=listener,
            #     sender=sender,
            #     a_share=a_share,
            #     b_share=b_share,
            #     total_bit=total_bit,
            #     msb_pos=pos,
            #     triple=triple
            # )
            ret = extract_msb_with_3_input_gate(role=role,
                                                listener=listener,
                                                sender=sender,
                                                a_share=a_share,
                                                b_share=b_share,
                                                total_bit=total_bit,
                                                msb_pos=pos,
                                                triple=triple,
                                                help_tuple=tuple)

            # Here ret_mo ^ ret_do = MSB

            # 3P - Oblivious Transfer

            if role == data_owner:
                ret = ot_3party(role=role,
                                sender=sender,
                                listener=listener,
                                random_handler=random_handler,
                                msg_dtype='u' + output_dtype,
                                msg_bit=output_bit,
                                choose_bit=ret)

            elif role == model_owner:
                msg_0 = ret
                msg_1 = True ^ ret
                msg_0 = ((True ^ msg_0) +
                         msg_0 * cur_layer['deact_val']).astype(output_dtype)
                msg_1 = ((True ^ msg_1) +
                         msg_1 * cur_layer['deact_val']).astype(output_dtype)
                msg_0 = int2uint(msg_0)
                msg_1 = int2uint(msg_1)

                # add r to two messages
                r = get_random_mask(msg_0.shape, bits=output_bit)
                msg_0 -= r
                msg_1 -= r

                # send the message
                ot_3party(role=role,
                          sender=sender,
                          listener=listener,
                          random_handler=random_handler,
                          msg_0=msg_0,
                          msg_1=msg_1,
                          msg_bit=output_bit)
                ret = r

            elif role == ttp:
                ot_3party(role=role,
                          sender=sender,
                          listener=listener,
                          random_handler=random_handler,
                          msg_dtype='u' + output_dtype,
                          msg_shape=ori_shape)
                ret = np.zeros(shape=ori_shape, dtype='u' + output_dtype)

            # Reshare
            if cur_layer['deact_val'] == -1:
                random_handler.step_synchronous()
                mask = random_handler.get_3_out_of_3_shares(output_bit,
                                                            size=ret.shape)
                ret += mask
                send_thread = sender[role_last].send_ndarray_by_bytes_thread(
                    ret)
                send_thread.start()
                share_rcv = listener[role_next].receive_ndarray_by_bytes(
                    size=ret.shape, dtype=ret.dtype)
                ret = tf.constant([ret, share_rcv], dtype=tf.int32)
                send_thread.join()
            else:
                ret = tf.constant([
                    ret,
                ], dtype=tf.int32)

        elif name.startswith('bnn__dense'):
            ret = tf.matmul(ret[0], cur_layer['W'][0]) + \
                tf.matmul(ret[0], cur_layer['W'][1]) + \
                tf.matmul(ret[1], cur_layer['W'][0])
            ret += random_handler.get_3_out_of_3_shares(bits=int(
                cur_layer['dtype'][3:]),
                size=ret.shape)
            random_handler.cnter += 1

        elif name.startswith('bnn__conv2d'):
            ret = tf.nn.conv2d(
                ret[0], cur_layer['W'][0], strides=cur_layer['strides'], padding=cur_layer['padding']) + \
                tf.nn.conv2d(
                    ret[0], cur_layer['W'][1], strides=cur_layer['strides'], padding=cur_layer['padding']) + \
                tf.nn.conv2d(
                    ret[1], cur_layer['W'][0], strides=cur_layer['strides'], padding=cur_layer['padding'])
            ret += random_handler.get_3_out_of_3_shares(bits=int(
                cur_layer['dtype'][3:]),
                size=ret.shape)
            random_handler.cnter += 1

        elif name.startswith('flatten'):
            share_0 = tf.reshape(ret[0], (ret[0].shape[0], -1))
            share_1 = tf.reshape(ret[1], (ret[1].shape[0], -1))
            ret = [share_0, share_1]

    return ret


def test_secure_activation(n_bit: int, n_batch: int, role: int, listener: list, sender: list, config):
    data_owner = 0
    model_owner = 1
    ttp = 2

    role_last = (role - 1) % 3
    role_next = (role + 1) % 3

    ret = np.zeros(n_batch, dtype='int32')
    cur_layer = {
        'dtype': 'int32',
        'output_dtype': 'int32',
        'msb': n_bit-1,
        'nxt_dtype': 'int32',
        'deact_val': -1
    }

    input_dtype = cur_layer['dtype']
    output_dtype = cur_layer['nxt_dtype']

    total_bit = 32
    input_bit = int(input_dtype[3:])
    pos = cur_layer['msb']

    output_bit = int(output_dtype[3:])
    ori_shape = ret.shape

    # n_triple = 3 * (pos - 1) + 1
    # n_triple = count_ppa_n_triple(msb_pos=pos)
    n_tuple, n_triple = count_tuple_and_triple(msb_pos=pos)

    random_handler = RandomHandler(role, config['r'][role],
                                   config['r'][role_next])
    # Prepare beaver's triple
    '''
    tuple:
    u,  v,  w,  uv, uw, vw, uvw
    0,  1,  2,   3,  4,  5, 6

    u,  v,  w1, uv, uw1,    vw1,    uvw1
    0,  1,  7,  3,  8,      9,      10

    u,  v2, uv2
    0,  11, 12

    generate tuple, then generate triple
    '''

    if role == ttp:
        send_datashare_thread = sender[
            data_owner].send_ndarray_by_bytes_thread(ret)
        send_datashare_thread.start()

        tuple_do = random_handler.get_same_random_with(
            data_owner, bits=1, size=(13, n_tuple, *ret.shape))
        tuple_mo = random_handler.get_same_random_with(
            model_owner, bits=1, size=(13, n_tuple, *ret.shape))
        random_handler.cnter += 1

        triple_do = random_handler.get_same_random_with(
            data_owner, bits=1, size=(3, n_triple, *ret.shape))
        triple_mo = random_handler.get_same_random_with(
            model_owner, bits=1, size=(3, n_triple, *ret.shape))
        random_handler.cnter += 1

        # delta = modify_triple_with_optimization(
        #     pos, triple_do, triple_mo)
        # delta = (triple_do[0] ^ triple_mo[0]) & (
        #     triple_do[1] ^ triple_mo[1]) ^ triple_do[2]

        delta_tuple = get_tuple_delta(tuple_mo, tuple_do)
        delta_triple = get_triple_delta(triple_mo, triple_do)
        delta = np.append(delta_tuple.reshape(-1),
                          delta_triple.reshape(-1))

        send_delta_thread = sender[
            model_owner].send_ndarray_by_bytes_thread(delta,
                                                      packbits=True)
        send_delta_thread.start()
        a_share = None
        b_share = None
        triple = None
        tuple = None
        send_delta_thread.join()

        send_datashare_thread.join()

    elif role == data_owner:
        ttp_share = listener[ttp].receive_ndarray_by_bytes(
            size=ret.shape, dtype=ret.dtype)
        ret += ttp_share

        tuple = random_handler.get_same_random_with(ttp,
                                                    bits=1,
                                                    size=(13, n_tuple,
                                                          *ret.shape))
        random_handler.cnter += 1

        triple = random_handler.get_same_random_with(ttp,
                                                     bits=1,
                                                     size=(3, n_triple,
                                                           *ret.shape))
        random_handler.cnter += 1

        a_share = ret
        b_share = np.zeros_like(ret)

    elif role == model_owner:

        tuple = random_handler.get_same_random_with(ttp,
                                                    bits=1,
                                                    size=(13, n_tuple,
                                                          *ret.shape))
        random_handler.cnter += 1

        triple = random_handler.get_same_random_with(ttp,
                                                     bits=1,
                                                     size=(3, n_triple,
                                                           *ret.shape))
        random_handler.cnter += 1

        delta = listener[ttp].receive_ndarray_by_bytes(
            size=((8 * n_tuple + n_triple) * np.prod(ret.shape)),
            packbits=True)

        len_delta_triple = np.prod(triple[2].shape)
        len_delta_tuple = len(delta) - len_delta_triple

        delta_tuple = None
        delta_triple = None

        if len_delta_tuple > 0:
            delta_tuple = np.reshape(delta[:len_delta_tuple],
                                     (-1, n_tuple, *ret.shape))
            tuple[3] = delta_tuple[0]
            tuple[4] = delta_tuple[1]
            tuple[5] = delta_tuple[2]
            tuple[6] = delta_tuple[3]
            tuple[8] = delta_tuple[4]
            tuple[9] = delta_tuple[5]
            tuple[10] = delta_tuple[6]
            tuple[12] = delta_tuple[7]

        if len_delta_triple > 0:
            delta_triple = np.reshape(delta[len_delta_tuple:],
                                      triple[2].shape)
            triple[2] = delta_triple

        a_share = np.zeros_like(ret)
        b_share = ret

    ret = extract_msb_with_3_input_gate(role=role,
                                        listener=listener,
                                        sender=sender,
                                        a_share=a_share,
                                        b_share=b_share,
                                        total_bit=total_bit,
                                        msb_pos=pos,
                                        triple=triple,
                                        help_tuple=tuple)

    # Here ret_mo ^ ret_do = MSB

    # 3P - Oblivious Transfer

    if role == data_owner:
        ret = ot_3party(role=role,
                        sender=sender,
                        listener=listener,
                        random_handler=random_handler,
                        msg_dtype='u' + output_dtype,
                        msg_bit=output_bit,
                        choose_bit=ret)

    elif role == model_owner:
        msg_0 = ret
        msg_1 = True ^ ret
        msg_0 = ((True ^ msg_0) +
                 msg_0 * cur_layer['deact_val']).astype(output_dtype)
        msg_1 = ((True ^ msg_1) +
                 msg_1 * cur_layer['deact_val']).astype(output_dtype)
        msg_0 = int2uint(msg_0)
        msg_1 = int2uint(msg_1)

        # add r to two messages
        r = get_random_mask(msg_0.shape, bits=output_bit)
        msg_0 -= r
        msg_1 -= r

        # send the message
        ot_3party(role=role,
                  sender=sender,
                  listener=listener,
                  random_handler=random_handler,
                  msg_0=msg_0,
                  msg_1=msg_1,
                  msg_bit=output_bit)
        ret = r

    elif role == ttp:
        ot_3party(role=role,
                  sender=sender,
                  listener=listener,
                  random_handler=random_handler,
                  msg_dtype='u' + output_dtype,
                  msg_shape=ori_shape)
        ret = np.zeros(shape=ori_shape, dtype='u' + output_dtype)

    # Reshare
    random_handler.step_synchronous()
    mask = random_handler.get_3_out_of_3_shares(output_bit,
                                                size=ret.shape)
    ret += mask
    send_thread = sender[role_last].send_ndarray_by_bytes_thread(
        ret)
    send_thread.start()
    share_rcv = listener[role_next].receive_ndarray_by_bytes(
        size=ret.shape, dtype=ret.dtype)
    ret = tf.constant([ret, share_rcv], dtype=tf.int32)
    send_thread.join()

    return None
