# Communicator.py
import pickle, struct, socket, logging, io, torch

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Communicator")

class Communicator(object):
    def __init__(self, index=0, ip_address="0.0.0.0"):
        self.index = index
        self.ip = ip_address
        self.sock = socket.socket()

    def send_msg(self, sock, msg):
        """统一发送消息（pickle序列化）"""
        try:
            msg_pickle = pickle.dumps(msg)
            sock.sendall(struct.pack(">I", len(msg_pickle)))
            sock.sendall(msg_pickle)
            logger.debug(f"{msg[0]} sent to {sock.getpeername()[0]}:{sock.getpeername()[1]}")
        except Exception as e:
            logger.error(f"send_msg error: {e}")

    def recv_msg(self, sock, expect_msg_type=None):
        import io, torch, pickle, struct, socket
        try:
            hdr = sock.recv(4)
            if not hdr or len(hdr) < 4:
                logger.error("recv_msg error: Socket closed by peer (no header).")
                return None
            msg_len = struct.unpack(">I", hdr)[0]

            raw = sock.recv(msg_len, socket.MSG_WAITALL)
            if not raw:
                logger.error("recv_msg error: Socket closed by peer (no payload).")
                return None

            # 先用 pickle.loads
            try:
                msg = pickle.loads(raw)
            except Exception as e1:
                # 若因 CUDA 反序列化失败，回退用 torch.load 强制 map 到 CPU
                try:
                    msg = torch.load(io.BytesIO(raw), map_location=torch.device("cpu"), weights_only=False)
                except Exception as e2:
                    logger.error("recv_msg error: %s", str(e2))
                    return None

            # 期望类型校验
            if expect_msg_type is not None and isinstance(msg, (list, tuple)) and len(msg) > 0:
                if msg[0] == 'Finish':
                    return msg
                if msg[0] != expect_msg_type:
                    logger.warning("Expected %s but received %s", expect_msg_type, msg[0])
                    # 不再抛异常，返回 None 让上层决定
                    return None
            return msg

        except Exception as e:
            logger.error("recv_msg error: %s", str(e))
            return None

