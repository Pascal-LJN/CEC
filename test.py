import time
from Communicator import Communicator

class Client(Communicator):
    def __init__(self, index, ip, server_ip, server_port):
        super().__init__(index=index, ip_address=ip)
        self.sock.connect((server_ip, server_port))
        print(f"[Client-{index}] Connected to Server at {server_ip}:{server_port}")

    def run(self):
        for i in range(10):
            msg = self.recv_msg(self.sock, 'MSG_FROM_SERVER')
            print(f"[Client-{self.index}] Received: {msg[1]}")
            self.send_msg(self.sock, ['MSG_FROM_CLIENT', f'Ack from Client-{self.index}, round {i+1}'])
            time.sleep(1)

if __name__ == '__main__':
    # 替换为 node1 或 masternx 对应的 IP 地址
    client = Client(index=2, ip='192.168.123.181', server_ip='192.168.123.140', server_port=5124)
    client.run()
