from utils.randomTool import *
from utils.coding import *
from utils.npTcp import npTcpReceiver, npTcpSender
from utils.datasets import load_dataset
from utils.thread import custom_Thread
from utils.bnnModels import *
from utils.bnnLayers import *
from utils.mpc import SecureBiNN

import numpy as np
import json
from time import time
import tensorflow as tf
from sklearn.metrics import accuracy_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    # role 0: data owner (share_0, share_1)
    # role 1: model owner (share_1, share_2)
    # role 2: trusted third party (share_2, share_0)
    data_owner = 0
    model_owner = 1
    ttp = 2

    # Read the config
    with open('config.json', 'r') as f:
        config = json.load(f)
    precision = config['precision']
    float_coded_dtype = config['float_coded_dtype']

    role = config['role']
    role_last = (role - 1) % 3
    role_next = (role + 1) % 3
    precision = config['precision']
    n_repeat = config['repeat']
    random_handler = RandomHandler(role, config['r'][role],
                                   config['r'][role_next])

    # Build tcp connection
    sender = [None, None, None]
    listener = [None, None, None]

    listener[role_next] = npTcpReceiver()
    listener[role_last] = npTcpReceiver()
    listener[role_next].bind_listen(
        (config['ip_{}'.format(role)],
         config['port_{}_{}'.format(role, role_next)]))
    listener[role_last].bind_listen(
        (config['ip_{}'.format(role)],
         config['port_{}_{}'.format(role, role_last)]))
    thread_listen_next = custom_Thread(target=listener[role_next].accept)
    thread_listen_last = custom_Thread(target=listener[role_last].accept)
    thread_listen_next.start()
    thread_listen_last.start()

    sender[role_next] = npTcpSender()
    sender[role_last] = npTcpSender()
    thread_send_next = custom_Thread(target=sender[role_next].connect,
                                     args=(config['ip_{}'.format(role_next)],
                                           config['port_{}_{}'.format(
                                               role_next, role)]))
    thread_send_last = custom_Thread(target=sender[role_last].connect,
                                     args=(config['ip_{}'.format(role_last)],
                                           config['port_{}_{}'.format(
                                               role_last, role)]))
    thread_send_last.start()
    thread_send_next.start()

    thread_listen_next.join()
    thread_listen_last.join()
    thread_send_last.join()
    thread_send_next.join()

    # seperate model
    print('Seperate model')
    model_share = None

    if role == model_owner:
        model = load_bnn_model(config['model_path'])
        list_layers = extract_layers_from_model(model)
        list_layers_coded = code_list_layers(
            list_layers,
            precision=precision,
            float_coded_dtype=float_coded_dtype)
        lst_model_shares = code_model_to_shares(list_layers_coded)

        share_do = np.array(lst_model_shares[data_owner], dtype=object)
        send_to_dataowner = custom_Thread(
            target=sender[data_owner].send_ndarray, args=(share_do, ))
        send_to_dataowner.start()

        share_ttp = np.array(lst_model_shares[ttp], dtype=object)
        send_to_ttp = custom_Thread(target=sender[ttp].send_ndarray,
                                    args=(share_ttp, ))
        send_to_ttp.start()

        model_share = np.array(lst_model_shares[model_owner], dtype=object)
        send_to_dataowner.join()
        send_to_ttp.join()

        del lst_model_shares, send_to_dataowner, send_to_ttp

    else:
        model_share = listener[model_owner].receive_ndarray()

    if role == model_owner:
        custom_model_summary(model_share)

    # convert numpy into tf.constant
    for i in range(len(model_share)):
        for key in model_share[i].keys():
            if key == 'W':
                model_share[i][key][0] = tf.constant(model_share[i][key][0],
                                                     dtype=tf.int32)
                model_share[i][key][1] = tf.constant(model_share[i][key][1],
                                                     dtype=tf.int32)
            elif key == 'b' or key == 'val':
                model_share[i][key] = tf.constant(model_share[i][key],
                                                  dtype=tf.int32)
            elif key == 'padding':
                model_share[i][key] = model_share[i][key].upper()

    # seperate data
    print('Seperate data')
    data_share = None
    y_ans = None
    if role == data_owner:
        # Load dataset
        flatten = False if model_share[0]['name'].find('conv') > -1 else True
        (X_train, y_train), (X_test, y_test) = load_dataset(config["dataset"],
                                                            flatten=flatten,
                                                            scale=True,
                                                            archive_path=config["archive_path"])
        del X_train, y_train

        # Select batch
        if config['eval_on_entire_test_set'] == True:
            X, y = X_test, y_test
            y_ans = y
        else:
            X, y = X_test[:config['batch_size']], y_test[:config['batch_size']]
            y_ans = y

        # Code and separate
        X_coded = float2udtype(X, float_coded_dtype, precision)
        lst_data_shares = split_into_n_picies(X_coded, n_piece=3)
        mo_share = np.array([
            lst_data_shares[model_owner],
            lst_data_shares[(model_owner + 1) % 3]
        ])
        ttp_share = np.array(
            [lst_data_shares[ttp], lst_data_shares[(ttp + 1) % 3]])

        send_to_model_owner = custom_Thread(sender[model_owner].send_ndarray,
                                            args=(mo_share, ))
        send_to_model_owner.start()
        send_to_ttp = custom_Thread(sender[ttp].send_ndarray,
                                    args=(ttp_share, ))
        send_to_ttp.start()

        data_share = np.array([
            lst_data_shares[data_owner], lst_data_shares[(data_owner + 1) % 3]
        ])

        send_to_model_owner.join()
        send_to_ttp.join()

        del X_coded, lst_data_shares, X, y, mo_share, ttp_share
    else:
        data_share = listener[data_owner].receive_ndarray()

    if role == data_owner:
        print('The shape of the answer:', y_ans.shape)

    # Convert to tf constant
    data_share[0] = uint2int(data_share[0])
    data_share[1] = uint2int(data_share[1])
    data_share = [
        tf.constant(data_share[0], dtype=tf.int32),
        tf.constant(data_share[1], dtype=tf.int32)
    ]
    n_data = data_share[0].shape[0]
    n_batch = config['batch_size']

    # Evaluate
    print('Evaluate begin:')
    tic = time()
    # Initialize
    sender[role_next].send_bytes = 0
    sender[role_last].send_bytes = 0
    listener[role_next].receive_bytes = 0
    listener[role_last].receive_bytes = 0
    random_handler.cnter = 0

    ret_list = list()

    for cnt_repeat in range(n_repeat):
        print('Iter {} begin.'.format(cnt_repeat))
        ret_list = list()
        for l in range(0, n_data, n_batch):
            r = min(n_data, l + n_batch)
            data_share_batch = [data_share[0][l:r], data_share[1][l:r]]
            ret = SecureBiNN(role, random_handler, listener, sender,
                             data_share_batch, model_share)
            ret_list.append(ret)
    toc = time()

    ret_list = np.array([item.numpy().reshape(-1)
                         for item in ret_list]).reshape(-1)

    len_rcv = listener[role_last].receive_bytes + \
        listener[role_next].receive_bytes


    if role != data_owner:
        sender[data_owner].send_ndarray(ret_list)
        sender[data_owner].send(str(len_rcv).encode())
    else:
        ret_mo = listener[model_owner].receive_ndarray()
        ret_ttp = listener[ttp].receive_ndarray()
        result = ret_mo + ret_ttp + ret_list
        result = result.reshape(y_ans.shape)

        result = udtype2float(result, float_coded_dtype, precision)
        result = result.reshape(y_ans.shape)
        acc = accuracy_score(np.argmax(y_ans, axis=1), np.argmax(result,
                                                                 axis=1))

        msglen_mo = int(listener[model_owner].receive().decode())
        msglen_ttp = int(listener[ttp].receive().decode())
        msglen_byte = msglen_mo + msglen_ttp + len_rcv

        msglen_mb = msglen_byte / (1024.0 * 1024.0)
        print('N_iter: {}'.format(n_repeat))
        print('Accuracy: {:.04f} %'.format(acc * 100))

        print('Total:')
        print('Time: {:.03f}s, Comm: {:.03f}MB'.format(
            toc - tic, msglen_mb))

        print('Per iteration:')
        print('Time: {:.03f}s, Comm: {:.03f}MB'.format(
            (toc - tic) / n_repeat, msglen_mb / n_repeat))

        print('Per input:')
        print('Time: {:.03f}s, Comm: {:.03f}MB'.format(
            (toc - tic) / n_repeat / n_data, msglen_mb / n_data / n_repeat))

    for item in listener:
        if item is not None:
            item.close()
    for item in sender:
        if item is not None:
            item.close()
