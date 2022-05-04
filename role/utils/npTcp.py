import socket
import pickle
import time
import numpy as np

# from tensorflow.python.ops.gen_array_ops import pack

from utils.thread import custom_Thread

_ACC_SIGN = 'A'
_REJ_SIGN = 'R'


class npTcpReceiver:

    def __init__(self, bufsize=4096):
        self.skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.default_bufsize = bufsize
        self._code_method = 'utf-8'
        self.skt_accepted = None
        self.add_sender = None
        self.skt.settimeout(None)
        self.receive_bytes = 0

    def bind(self, add):
        self.skt.bind(add)

    def listen(self, backlog=1):
        self.skt.listen(backlog)

    def bind_listen(self, add, backlog=1):
        self.skt.bind(add)
        self.skt.listen(backlog)

    def accept(self):
        self.skt_accepted, self.add_sender = self.skt.accept()

    def send(self, msg):
        self.skt_accepted.send(msg)

    def receive(self, bufsize=None):
        if bufsize is None:
            bufsize = self.default_bufsize
        return self.skt_accepted.recv(bufsize)

    def recv_into(self, buffer):
        self.skt_accepted.recv_into(buffer)

    def receive_ndarray(self, bufsize=None):
        if bufsize is None:
            bufsize = self.default_bufsize

        # confirm the length
        msg_len = int(self.skt_accepted.recv(
            bufsize).decode(self._code_method))
        self.skt_accepted.send(str(msg_len).encode())
        # print(msg_len)

        # receive the bytes
        chunk = b''
        rcv_len = 0
        while rcv_len < msg_len:
            tmp = self.skt_accepted.recv(bufsize)
            chunk += tmp
            rcv_len += len(tmp)

        # self.skt_accepted.recv_into(chunk)
        arr = pickle.loads(chunk)

        # confirm accepted
        self.skt_accepted.send(_ACC_SIGN.encode())
        self.receive_bytes += msg_len
        return arr

    def receive_ndarray_by_bytes(self, size, dtype=None, packbits=False, bufsize=None):
        if bufsize is None:
            bufsize = self.default_bufsize
        if packbits == True:
            if dtype is not None:
                assert dtype == 'uint8', 'dtype must be uint8 if "packbits" is true !'
            else:
                dtype = 'uint8'
        if packbits == False and dtype == None:
            raise ''

        msg_len = None
        if packbits == True:
            msg_len = (np.prod(size) + 7) // 8
        else:
            msg_len = len(np.array([0], dtype=dtype).tobytes()) * np.prod(size)

        # receive the bytes
        chunk = b''
        rcv_len = 0
        while rcv_len < msg_len:
            tmp = self.skt_accepted.recv(bufsize)
            chunk += tmp
            rcv_len += len(tmp)

        # unpack the bits when transfer booleans
        if packbits == True:
            dtype = 'uint8'
        arr = np.frombuffer(chunk, dtype=dtype)

        if packbits == True:
            arr = np.unpackbits(arr).astype('bool')
            trunc_len = np.prod(size)
            if msg_len - trunc_len < 8:
                arr = arr[:trunc_len]

        if size is not None:
            arr = arr.reshape(size)

        # confirm accepted
        self.skt_accepted.send(_ACC_SIGN.encode())
        self.receive_bytes += msg_len
        return arr

    def receive_ndarray_thread(self, bufsize=None):
        ret = custom_Thread(
            target=self.receive_ndarray, args=(bufsize,))
        return ret

    def receive_ndarray_by_bytes_thread(self, size, dtype=None, packbits=False, bufsize=None):
        ret = custom_Thread(
            target=self.receive_ndarray_by_bytes, args=(size, dtype, packbits, bufsize))
        return ret

    def close(self):
        self.skt.close()


class npTcpSender:

    def __init__(self, bufsize=4096):
        self.skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.default_bufsize = bufsize
        self._code_method = 'utf-8'
        self.ip_rcver = None
        self.port_rcver = None
        self.skt.settimeout(None)
        self.send_bytes = 0

    def connect(self, ip, port):
        self.ip_rcver = ip
        self.port_rcver = port

        while True:
            try:
                self.skt.connect((ip, port))
                print('Connect to {}:{} successfully.'.format(ip, port))
                break
            except(ConnectionError):
                time.sleep(0.1)

    def send(self, msg):
        self.skt.send(msg)

    def receive(self, bufsize=None):
        if bufsize is None:
            bufsize = self.default_bufsize
        return self.skt.recv(bufsize)

    def recv_into(self, buffer):
        self.skt.recv_into(buffer)

    def send_ndarray(self, arr: np.ndarray, bufsize=None):
        if bufsize is None:
            bufsize = self.default_bufsize

        # send the length
        arr_dumped = pickle.dumps(arr)
        msg_len = len(arr_dumped)
        self.skt.send(str(msg_len).encode(self._code_method))
        # print(msg_len, 'to sent')
        # confirm msg_len
        tmp = int(self.skt.recv(bufsize).decode(self._code_method))
        if tmp != msg_len:
            raise ConnectionError(
                '{}:{} did not receive the message length correctly.'.format(self.ip_rcver, self.port_rcver))

        # send the bytes
        total_sent = 0
        while total_sent < msg_len:
            cur_sent = self.skt.send(arr_dumped[total_sent:])
            total_sent += cur_sent
        self.send_bytes += total_sent

        # confirm
        ans = self.skt.recv(bufsize).decode(self._code_method)
        if ans != _ACC_SIGN:
            raise ConnectionError('{}:{} did not send ACC SIGN.'.format(
                self.ip_rcver, self.port_rcver))

    def send_ndarray_by_bytes(self, arr: np.ndarray, packbits=False, bufsize=None):
        if bufsize is None:
            bufsize = self.default_bufsize

        if packbits == True:
            assert arr.dtype == 'bool' or arr.dtype == np.bool, 'Dtype {} is not supported.'.format(
                arr.dtype)

        # send the length
        arr_flatten = arr.reshape(-1)

        if packbits == True:
            arr_flatten = np.packbits(arr_flatten)

        arr_bytes = arr_flatten.tobytes()
        msg_len = len(arr_bytes)
        # self.skt.send(str(msg_len).encode(self._code_method))

        # confirm msg_len
        # tmp = int(self.skt.recv(bufsize).decode(self._code_method))
        # if tmp != msg_len:
        #     raise ConnectionError(
        #         '{}:{} did not receive the message length correctly.'.format(self.ip_rcver, self.port_rcver))

        # send the bytes
        total_sent = 0
        while total_sent < msg_len:
            cur_sent = self.skt.send(arr_bytes[total_sent:])
            total_sent += cur_sent
        self.send_bytes += total_sent

        ans = self.skt.recv(bufsize).decode(self._code_method)
        if ans != _ACC_SIGN:
            raise ConnectionError('{}:{} did not send ACC SIGN.'.format(
                self.ip_rcver, self.port_rcver))

        return

    def send_ndarray_thread(self, arr: np.ndarray, bufsize=None):
        ret = custom_Thread(target=self.send_ndarray, args=(arr, bufsize))
        return ret

    def send_ndarray_by_bytes_thread(self, arr: np.ndarray, packbits=False, bufsize=None):
        ret = custom_Thread(
            target=self.send_ndarray_by_bytes, args=(arr, packbits, bufsize))
        return ret

    def close(self):
        self.skt.close()
