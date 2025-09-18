# Communicator Object

import pickle
import struct
import socket
import io

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # 如果没有 torch，也能运行

class Communicator(object):
    def __init__(self, index, ip_address):
        self.index = index
        self.ip = ip_address
        self.sock = socket.socket()

    def send_msg(self, sock, msg):
        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)
        logger.debug(msg[0]+' sent to '+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

    def recv_msg(self, sock, expect_msg_type=None):
        msg_len = struct.unpack(">I", sock.recv(4))[0]
        raw = sock.recv(msg_len, socket.MSG_WAITALL)

        # 默认用 pickle 反序列化
        msg = pickle.loads(raw)

        # 如果 torch 可用，尝试强制 map 到 CPU
        if torch is not None:
            def to_cpu(obj):
                try:
                    if isinstance(obj, (bytes, bytearray)):
                        return torch.load(io.BytesIO(obj), map_location=torch.device('cpu'))
                    elif isinstance(obj, dict):
                        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in obj.items()}
                    elif torch.is_tensor(obj):
                        return obj.cpu()
                except Exception:
                    pass
                return obj

            # 遍历 msg 内容，把 GPU tensor 映射到 CPU
            if isinstance(msg, (list, tuple)):
                msg = list(msg)
                for i in range(len(msg)):
                    msg[i] = to_cpu(msg[i])

        logger.debug(msg[0]+' received from '+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg
