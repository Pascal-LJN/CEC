import torch
import torch.nn as nn
import torch.optim as optim

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import tqdm
import time
import numpy as np
import threading
import json
import operator

import sys
sys.path.append('../')
import config
import utils
from Communicator import *

# -----------------------------
# 关键：统一提供 CPU 权重 dict
# -----------------------------
def cpu_state_dict(sd: dict):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

if config.random:
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    logger.info('Random seed: {}'.format(config.random_seed))


class Env(Communicator):
    def __init__(self, index, ip_address, server_port, clients_list, model_name, model_cfg, batchsize):
        super(Env, self).__init__(index, ip_address)
        self.index = index
        self.clients_list = clients_list
        self.model_name = model_name
        self.batchsize = batchsize
        self.model_cfg = model_cfg
        self.state_dim = 2 * config.G
        self.action_dim = config.G
        self.group_labels = []
        self.model_flops_list = self.get_model_flops_list(model_cfg, model_name)
        assert len(self.model_flops_list) == config.model_len

        # Server configration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        self.model_name = model_name
        self.sock.bind((self.ip, self.port))
        self.client_socks = {}

        while len(self.client_socks) < config.K:
            self.sock.listen(5)
            (client_sock, (ip, port)) = self.sock.accept()
            self.client_socks[str(ip)] = client_sock

        self.uninet = utils.get_model('Unit', self.model_name, 0, self.device, self.model_cfg)
        self.criterion = nn.CrossEntropyLoss()

    def reset(self, done, first):
        split_layers = [config.model_len - 1 for _ in range(config.K)]  # no offloading
        config.split_layer = split_layers
        thread_number = config.K
        client_ips = config.CLIENTS_LIST
        self.initialize(split_layers)
        self.scatter(['RESET_FLAG', True])

        # Test network speed
        self.test_network(thread_number, client_ips)
        self.network_state = {}
        for s in self.client_socks:
            msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK_SPEED')
            if msg is None:
                logger.error("Network speed reply None from %s", s)
                continue
            self.network_state[msg[1]] = msg[2]

        # Classic FL warmup
        if first:
            self.infer(thread_number, client_ips)
            self.infer(thread_number, client_ips)
        else:
            self.infer(thread_number, client_ips)

        self.offloading_state = self.get_offloading_state(
            split_layers, self.clients_list, self.model_cfg, self.model_name
        )
        self.baseline = self.infer_state
        if len(self.group_labels) == 0:
            self.group_model, self.cluster_centers, self.group_labels = self.group(self.baseline, self.network_state)

        logger.info('Basiline: ' + json.dumps(self.baseline))

        state = self.concat_norm(self.clients_list, self.network_state, self.infer_state, self.offloading_state)
        assert self.state_dim == len(state)
        return np.array(state)

    def group(self, baseline, network):
        from sklearn.cluster import KMeans
        X = []
        for c in self.clients_list:
            X.append([baseline[c]])

        kmeans = KMeans(n_clusters=config.G, random_state=0).fit(X)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.predict(X)
        return kmeans, cluster_centers, labels

    def step(self, action, done):
        # Expand action to each device and initialization
        action = self.expand_actions(action, self.clients_list)
        config.split_layer = self.action_to_layer(action)
        split_layers = config.split_layer
        logger.info('Current OPs: ' + str(split_layers))
        thread_number = config.K
        client_ips = config.CLIENTS_LIST
        self.initialize(split_layers)

        self.scatter(['RESET_FLAG', False])
        self.scatter(['SPLIT_LAYERS', config.split_layer])

        # Test network speed
        self.test_network(thread_number, client_ips)
        self.network_state = {}
        for s in self.client_socks:
            msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK_SPEED')
            if msg is None:
                logger.error("Network speed reply None from %s", s)
                continue
            self.network_state[msg[1]] = msg[2]

        # Offloading training and return env state
        self.infer(thread_number, client_ips)
        self.offloading_state = self.get_offloading_state(
            split_layers, self.clients_list, self.model_cfg, self.model_name
        )
        reward, maxtime, done = self.calculate_reward(self.infer_state)
        logger.info('Training time per iteration: ' + json.dumps(self.infer_state))
        state = self.concat_norm(self.clients_list, self.network_state, self.infer_state, self.offloading_state)
        assert self.state_dim == len(state)

        return np.array(state), reward, maxtime, done

    def initialize(self, split_layers):
        self.split_layers = split_layers
        self.nets = {}
        self.optimizers = {}
        for i in range(len(split_layers)):
            client_ip = config.CLIENTS_LIST[i]
            if split_layers[i] < config.model_len - 1:  # Only offloading client need initialized in server
                self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, self.model_cfg)
                self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=config.LR, momentum=0.9)

    def test_network(self, thread_number, client_ips):
        self.net_threads = {}
        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]] = threading.Thread(
                target=self._thread_network_testing, args=(client_ips[i],)
            )
            self.net_threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]].join()

    def _thread_network_testing(self, client_ip):
        msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK_SPEED')
        if msg is None:
            return
        # 关键：回 CPU 权重（避免 CUDA 反序列化）
        reply = ['MSG_TEST_NETWORK_SPEED', cpu_state_dict(self.uninet.cpu().state_dict())]
        self.send_msg(self.client_socks[client_ip], reply)

    def infer(self, thread_number, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            if self.split_layers[i] == config.model_len - 1:
                self.threads[client_ips[i]] = threading.Thread(
                    target=self._thread_infer_no_offloading, args=(client_ips[i],)
                )
                self.threads[client_ips[i]].start()
            else:
                self.threads[client_ips[i]] = threading.Thread(
                    target=self._thread_infer_offloading, args=(client_ips[i],)
                )
                self.threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

        self.infer_state = {}
        for s in self.client_socks:
            msg = self.recv_msg(self.client_socks[s], 'MSG_INFER_SPEED')
            if msg is None:
                logger.error("Infer speed reply None from %s", s)
                continue
            self.infer_state[msg[1]] = msg[2]

    def _thread_infer_no_offloading(self, client_ip):
        pass

    def _thread_infer_offloading(self, client_ip):
        for _ in range(config.iteration[client_ip]):
            msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
            if msg is None:
                return

            # 1) 收到 smashed_layers / labels
            smashed_layers = msg[1]
            labels = msg[2]

            # 2) 放到 server 设备，并把 smashed_layers 当作叶子变量、开启梯度
            inputs = smashed_layers.to(self.device, non_blocking=False).detach().requires_grad_(True)
            targets = labels.to(self.device, non_blocking=False)

            # 3) 前向 + 反向
            self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()  # 会计算 dLoss/dInputs
            self.optimizers[client_ip].step()

            # 4) 把 dLoss/dInputs 回传给客户端（转 CPU，保持连续）
            grad_cpu = inputs.grad.detach().to('cpu', non_blocking=False).contiguous()
            reply = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_' + str(client_ip), grad_cpu]
            self.send_msg(self.client_socks[client_ip], reply)

    def scatter(self, msg):
        for i in self.client_socks:
            self.send_msg(self.client_socks[i], msg)

    def get_offloading_state(self, split_layer, clients_list, model_cfg, model_name):
        offloading_state = {}
        offload = 0
        assert len(split_layer) == len(clients_list)
        for i in range(len(clients_list)):
            for l in range(len(model_cfg[model_name])):
                if l <= split_layer[i]:
                    offload += model_cfg[model_name][l][5]
            offloading_state[clients_list[i]] = offload / config.total_flops
            offload = 0
        return offloading_state

    def get_model_flops_list(self, model_cfg, model_name):
        model_state_flops = []
        cumulated_flops = 0
        for l in model_cfg[model_name]:
            cumulated_flops += l[5]
            model_state_flops.append(cumulated_flops)
        model_flops_list = np.array(model_state_flops)
        model_flops_list = model_flops_list / cumulated_flops
        return model_flops_list

    def concat_norm(self, clients_list, network_state, infer_state, offloading_state):
        network_state_order = []
        infer_state_order = []
        offloading_state_order = []
        for c in clients_list:
            network_state_order.append(network_state[c])
            infer_state_order.append(infer_state[c])
            offloading_state_order.append(offloading_state[c])

        group_max_index = [0 for _ in range(config.G)]
        group_max_value = [0 for _ in range(config.G)]
        for i in range(len(clients_list)):
            label = self.group_labels[i]
            if infer_state_order[i] >= group_max_value[label]:
                group_max_value[label] = infer_state_order[i]
                group_max_index[label] = i

        infer_state_order = np.array(infer_state_order)[np.array(group_max_index)]
        offloading_state_order = np.array(offloading_state_order)[np.array(group_max_index)]
        network_state_order = np.array(network_state_order)[np.array(group_max_index)]
        state = np.append(infer_state_order, offloading_state_order)
        return state

    def calculate_reward(self, infer_state):
        reward = 0
        done = False
        max_basetime = max(self.baseline.items(), key=operator.itemgetter(1))[1]
        max_infertime = max(infer_state.items(), key=operator.itemgetter(1))[1]

        if max_infertime >= 1 * max_basetime:
            done = True
        else:
            done = False

        for k in infer_state:
            if infer_state[k] < self.baseline[k]:
                r = (self.baseline[k] - infer_state[k]) / self.baseline[k]
                reward += r
            else:
                r = (infer_state[k] - self.baseline[k]) / infer_state[k]
                reward -= r

        return reward, max_infertime, done

    def expand_actions(self, actions, clients_list):
        full_actions = []
        for i in range(len(clients_list)):
            full_actions.append(actions[self.group_labels[i]])
        return full_actions

    def action_to_layer(self, action):
        split_layer = []
        for v in action:
            idx = np.where(np.abs(self.model_flops_list - v) == np.abs(self.model_flops_list - v).min())
            idx = idx[0][-1]
            if idx >= 5:  # all FC layers combine to one option
                idx = 6
            split_layer.append(idx)
        return split_layer


class RL_Client(Communicator):
    def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, split_layer, model_cfg):
        super(RL_Client, self).__init__(index, ip_address)
        self.ip_address = ip_address
        self.ip = ip_address  # 与服务端日志里取 msg[1] 一致
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.uninet = utils.get_model('Unit', self.model_name, 0, self.device, self.model_cfg)

        logger.info('==> Connecting to Server..')
        self.sock.connect((server_addr, server_port))

    def initialize(self, split_layer):
        self.split_layer = split_layer
        self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, self.model_cfg)
        self.optimizer = optim.SGD(self.net.parameters(), lr=config.LR, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        # First test network speed (双向发 CPU 权重)
        network_time_start = time.time()
        self.send_msg(self.sock, ['MSG_TEST_NETWORK_SPEED', cpu_state_dict(self.uninet.cpu().state_dict())])
        reply = self.recv_msg(self.sock, 'MSG_TEST_NETWORK_SPEED')
        if reply is None:
            logger.error("Network test failed: no reply")
            return
        network_time_end = time.time()
        network_speed = (2 * config.model_size * 8) / (network_time_end - network_time_start)  # Mbit/s
        self.send_msg(self.sock, ['MSG_TEST_NETWORK_SPEED', self.ip, network_speed])

    def infer(self, trainloader):
        self.net.to(self.device)
        self.net.train()
        s_time_infer = time.time()

        if self.split_layer == len(config.model_cfg[self.model_name]) - 1:  # No offloading
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if batch_idx >= config.iteration[self.ip_address] - 1:
                    break
        else:  # Offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                # 发送激活/标签到服务端 —— 都转 CPU
                self.send_msg(self.sock, ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER',
                                          outputs.detach().cpu(), targets.detach().cpu()])

                # 等待服务端梯度：带上期望类型，且判空
                expect = 'MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_' + str(self.ip_address)
                reply = self.recv_msg(self.sock, expect)
                if reply is None:
                    logger.error("No gradients from server, stop this round.")
                    return

                gradients = reply[1].to(self.device)
                outputs.backward(gradients)
                self.optimizer.step()

                if batch_idx >= config.iteration[self.ip_address] - 1:
                    break

        e_time_infer = time.time()
        logger.info('Training time: ' + str(e_time_infer - s_time_infer))

        infer_speed = (e_time_infer - s_time_infer) / config.iteration[self.ip_address]
        self.send_msg(self.sock, ['MSG_INFER_SPEED', self.ip, infer_speed])

    def reinitialize(self, split_layers):
        self.initialize(split_layers)
