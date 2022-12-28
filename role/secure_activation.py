import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.randomTool import *
from utils.coding import *
from utils.npTcp import npTcpReceiver, npTcpSender
from utils.thread import custom_Thread
from utils.bnnModels import *
from utils.bnnLayers import *
from utils.mpc import ot_3party, extract_msb_ppa, count_ppa_n_triple, modify_triple_with_optimization, extract_msb_ppa_with_optimize, verify_triple, SecureBiNN, test_secure_activation
import numpy as np
import json
from time import time
import tensorflow as tf
from sklearn.metrics import accuracy_score
from copy import deepcopy



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

    # Evaluate
    print('Secure Acitivation Evaluate begin:')

    time_list = list()
    comm_list = list()

    for n_bit in range(2, 33):
        tic = time()
        # Initialize
        sender[role_next].send_bytes = 0
        sender[role_last].send_bytes = 0
        listener[role_next].receive_bytes = 0
        listener[role_last].receive_bytes = 0
        random_handler.cnter = 0

        n_batch = 100000
        ret_list = list()

        for cnt_repeat in range(n_repeat):
            print('Iter {} begin.'.format(cnt_repeat))
            ret_list = list()
            test_secure_activation(
                n_bit=n_bit, n_batch=n_batch, role=role, listener=listener, sender=sender, config=config)
        toc = time()

        print('N_iter:{}'.format(n_repeat))
        print('Using time: {:.3f}s'.format(toc - tic))
        print('AVE time per iteration: {:.3f}s'.format((toc - tic) / n_repeat))
        print('AVE time per input: {:.3f}'.format((toc - tic) / n_repeat / n_batch))

        time_list.append((toc - tic) / n_repeat / n_batch)

        len_rcv = listener[role_last].receive_bytes + \
            listener[role_next].receive_bytes

        if role != data_owner:
            sender[data_owner].send(str(len_rcv).encode())
        else:
            msglen_mo = int(listener[model_owner].receive().decode())
            msglen_ttp = int(listener[ttp].receive().decode())
            msglen = msglen_mo + msglen_ttp + len_rcv
            print('Communication cost: {:.3f}MB'.format(msglen / 1024.0**2))
            print('Communication cost per iteration: {:.3f}MB'.format(
                msglen / 1024.0**2 / n_repeat))
            print('Communication cost per input: {:.3f}MB'.format(
                msglen / 1024.0**2 / n_batch / n_repeat))
            comm_list.append(msglen / 1024.0**2 / n_batch / n_repeat)

    for item in listener:
        if item is not None:
            item.close()
    for item in sender:
        if item is not None:
            item.close()

    if role == data_owner:
        time_list = np.array(time_list)
        comm_list = np.array(comm_list)
        if os.path.exists('result') == False:
            os.makedirs('result')
        np.save('result/eval_SAF_time.npy', time_list)
        np.save('result/eval_SAF_comm.npy', comm_list)
