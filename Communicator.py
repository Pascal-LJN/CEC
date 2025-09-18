# Communicator Object

import pickle
import struct
import socket
import torch

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
    def __init__(self, index, ip_address):
        self.index = index
        self.ip = ip_address
        self.sock = socket.socket()

    def send_msg(self, sock, msg):
        """序列化并发送消息"""
        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)
        logger.debug(msg[0] + ' sent to ' +
                     str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

    def recv_msg(self, sock, expect_msg_type=None):
        """接收并反序列化消息，自动 map 到 CPU"""
        msg_len = struct.unpack(">I", sock.recv(4))[0]
        raw = sock.recv(msg_len, socket.MSG_WAITALL)

        try:
            # GPU → CPU fallback
            msg = pickle.loads(raw, fix_imports=True,
                               encoding="latin1")
            # 如果 msg[1] 是模型权重 dict，确保落在 CPU
            if isinstance(msg, (list, tuple)) and len(msg) > 1 and isinstance(msg[1], dict):
                msg[1] = {k: (v.cpu() if torch.is_tensor(v) else v)
                          for k, v in msg[1].items()}
            # 如果是 tensor，也强制转 CPU
            elif isinstance(msg, (list, tuple)) and len(msg) > 1 and torch.is_tensor(msg[1]):
                msg[1] = msg[1].cpu()
        except Exception:
            # 最保险：直接强制 torch.load map 到 CPU
            import io
            msg = pickle.loads(raw, fix_imports=True,
                               encoding="latin1",
                               buffers=None)
            if torch.is_tensor(msg):
                msg = msg.cpu()

        logger.debug(msg[0] + ' received from ' +
                     str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception(
                    "Expected " + expect_msg_type + " but received " + msg[0])
        return msg
