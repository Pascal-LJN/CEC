# Sever.py  —— FL 服务器端（兼容 CPU 客户端）
import sys
import time
import json
import threading
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('../')
from Communicator import *  # noqa: E402
import utils  # noqa: E402
import config  # noqa: E402


def cpu_state_dict(sd: dict):
    """将 state_dict 的所有张量转为 CPU（用于跨设备发送）。"""
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}


class Sever(Communicator):
    def __init__(self, index, ip_address, server_port, model_name):
        super(Sever, self).__init__(index, ip_address)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        self.model_name = model_name

        # 监听 socket，等待 K 个客户端
        self.sock.bind((self.ip, self.port))
        self.client_socks = {}
        while len(self.client_socks) < config.K:
            self.sock.listen(5)
            logger.info("Waiting Incoming Connections.")
            (client_sock, (ip, port)) = self.sock.accept()
            logger.info('Got connection from %s', str(ip))
            logger.info(client_sock)
            self.client_socks[str(ip)] = client_sock

        # 全局模型放在 server 的 device 上
        self.uninet = utils.get_model(
            'Unit', self.model_name, config.model_len - 1, self.device, config.model_cfg
        )

        # 测试集（仅在服务端）
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.testset = torchvision.datasets.CIFAR10(
            root=config.dataset_path, train=False, download=True, transform=self.transform_test
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=2
        )

    def initialize(self, split_layers, offload, first, LR):
        """
        初始化每个客户端在服务器侧的分割模型与优化器。
        并下发全局权重（强制 CPU 以兼容 CPU 客户端）。
        """
        if offload or first:
            self.split_layers = split_layers
            self.nets = {}
            self.optimizers = {}
            for i in range(len(split_layers)):
                client_ip = config.CLIENTS_LIST[i]
                if split_layers[i] < len(config.model_cfg[self.model_name]) - 1:
                    # 该客户端会把前半段放本地，后半段在 server 上跑
                    self.nets[client_ip] = utils.get_model(
                        'Server', self.model_name, split_layers[i], self.device, config.model_cfg
                    )

                    # 保证 server 段和 client 段来源于同一套全局权重
                    cweights = utils.get_model(
                        'Client', self.model_name, split_layers[i], self.device, config.model_cfg
                    ).state_dict()
                    pweights = utils.split_weights_server(
                        self.uninet.state_dict(), cweights, self.nets[client_ip].state_dict()
                    )
                    self.nets[client_ip].load_state_dict(pweights)

                    self.optimizers[client_ip] = optim.SGD(
                        self.nets[client_ip].parameters(), lr=LR, momentum=0.9
                    )
                else:
                    # 不卸载（no offload）的占位网络
                    self.nets[client_ip] = utils.get_model(
                        'Server', self.model_name, split_layers[i], self.device, config.model_cfg
                    )
            self.criterion = nn.CrossEntropyLoss()

        # 下发全局初始参数 —— 一律转 CPU 再发
        init_msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT',
                    cpu_state_dict(self.uninet.state_dict())]
        for ip, sock in self.client_socks.items():
            self.send_msg(sock, init_msg)

    def train(self, thread_number, client_ips):
        # 1) 网络测试
        self.net_threads = {}
        for ip in client_ips:
            self.net_threads[ip] = threading.Thread(
                target=self._thread_network_testing, args=(ip,))
            self.net_threads[ip].start()
        for ip in client_ips:
            self.net_threads[ip].join()

        self.bandwidth = {}
        for ip in self.client_socks:
            msg = self.recv_msg(self.client_socks[ip], 'MSG_TEST_NETWORK')
            self.bandwidth[msg[1]] = msg[2]

        # 2) 开始训练
        self.threads = {}
        for i, ip in enumerate(client_ips):
            if config.split_layer[i] == (config.model_len - 1):
                # 不卸载（本地全量训练，server 无线程）
                self.threads[ip] = threading.Thread(
                    target=self._thread_training_no_offloading, args=(ip,))
                logger.info('%s no offloading training start', str(ip))
                self.threads[ip].start()
            else:
                # 卸载训练
                self.threads[ip] = threading.Thread(
                    target=self._thread_training_offloading, args=(ip,))
                logger.info('%s offloading training start', str(ip))
                self.threads[ip].start()

        for ip in client_ips:
            self.threads[ip].join()

        # 3) 回收统计
        self.ttpi = {}
        for ip in self.client_socks:
            msg = self.recv_msg(self.client_socks[ip], 'MSG_TRAINING_TIME_PER_ITERATION')
            self.ttpi[msg[1]] = msg[2]

        self.group_labels = self.clustering(self.ttpi, self.bandwidth)
        self.offloading = self.get_offloading(self.split_layers)
        state = self.concat_norm(self.ttpi, self.offloading)
        return state, self.bandwidth

    def _thread_network_testing(self, client_ip):
        # 客户端先发 'MSG_TEST_NETWORK'，这里回一个包含模型大小的包（权重用 CPU 版）
        _ = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
        reply = ['MSG_TEST_NETWORK', cpu_state_dict(self.uninet.state_dict())]
        self.send_msg(self.client_socks[client_ip], reply)

    def _thread_training_no_offloading(self, client_ip):
        # 纯本地训练，server 侧不需要干活
        pass

    def _thread_training_offloading(self, client_ip):
        iteration = int((config.N / (config.K * config.B)))
        for _ in range(iteration):
            # 接收客户端的 smashed layers（CPU 发送），搬到 server.device 上
            msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
            smashed_layers = msg[1].to(self.device, non_blocking=False)
            labels = msg[2].to(self.device, non_blocking=False)

            self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](smashed_layers)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizers[client_ip].step()

            # 把对 smashed_layers 的梯度发回客户端 —— 一律 CPU
            grad_cpu = smashed_layers.grad.detach().to('cpu', non_blocking=False).contiguous()
            reply = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_' + str(client_ip), grad_cpu]
            self.send_msg(self.client_socks[client_ip], reply)

        logger.info('%s offloading training end', str(client_ip))
        return 'Finish'

    def aggregate(self, client_ips):
        """
        从各客户端收集本地权重，统一搬到 self.device 上做 FedAvg，
        然后更新 self.uninet，并广播 'MSG_ROUND_DONE' 告知客户端一轮结束。
        """
        w_local_list = []
        for i, ip in enumerate(client_ips):
            msg = self.recv_msg(self.client_socks[ip], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
            local_weights = msg[1]

            # 把客户端上来的权重搬到 server.device
            local_weights = {k: v.to(self.device) for k, v in local_weights.items()}

            if config.split_layer[i] != (config.model_len - 1):
                merged = utils.concat_weights(
                    self.uninet.state_dict(), local_weights, self.nets[ip].state_dict()
                )
                w_local = (merged, config.N / config.K)
            else:
                w_local = (local_weights, config.N / config.K)

            w_local_list.append(w_local)

        # zero_model 也要搬到同一 device
        zero_model = {k: v.to(self.device) for k, v in utils.zero_init(self.uninet).state_dict().items()}
        aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)

        # 更新全局模型（在 server.device 上）
        self.uninet.load_state_dict(aggregrated_model)

        # 通知所有客户端：本轮完成（客户端在等这个）
        done_msg = ['MSG_ROUND_DONE']
        for ip, sock in self.client_socks.items():
            self.send_msg(sock, done_msg)

        # 返回 CPU 版权重（若上层需要）
        return cpu_state_dict(self.uninet.state_dict())

    def test(self, r):
        self.uninet.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.uninet(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        logger.info('Test Accuracy: %.2f', acc)

        # 保存 CPU 版 checkpoint（更通用）
        torch.save(cpu_state_dict(self.uninet.state_dict()), './' + config.model_name + '.pth')
        return acc

    def clustering(self, state, bandwidth):
        # 维持你原先的逻辑
        labels = [0, 0, 1, 0, 0]
        return labels

    def adaptive_offload(self, agent, state):
        action = agent.exploit(state)
        action = self.expand_actions(action, config.CLIENTS_LIST)
        config.split_layer = self.action_to_layer(action)
        logger.info('Next Round OPs: %s', str(config.split_layer))
        msg = ['SPLIT_LAYERS', config.split_layer]
        self.scatter(msg)
        return config.split_layer

    def expand_actions(self, actions, clients_list):
        return [actions[self.group_labels[i]] for i in range(len(clients_list))]

    def action_to_layer(self, action):
        model_state_flops = []
        cumulated_flops = 0
        for l in config.model_cfg[config.model_name]:
            cumulated_flops += l[5]
            model_state_flops.append(cumulated_flops)
        model_flops_list = np.array(model_state_flops) / cumulated_flops

        split_layer = []
        for v in action:
            idx = np.where(np.abs(model_flops_list - v) == np.abs(model_flops_list - v).min())[0][-1]
            if idx >= 5:  # FC 合并
                idx = 6
            split_layer.append(idx)
        return split_layer

    def concat_norm(self, ttpi, offloading):
        ttpi_order = []
        offloading_order = []
        for c in config.CLIENTS_LIST:
            ttpi_order.append(ttpi[c])
            offloading_order.append(offloading[c])

        group_max_index = [0 for _ in range(config.G)]
        group_max_value = [0 for _ in range(config.G)]
        for i in range(len(config.CLIENTS_LIST)):
            label = self.group_labels[i]
            if ttpi_order[i] >= group_max_value[label]:
                group_max_value[label] = ttpi_order[i]
                group_max_index[label] = i

        ttpi_order = np.array(ttpi_order)[np.array(group_max_index)]
        offloading_order = np.array(offloading_order)[np.array(group_max_index)]
        state = np.append(ttpi_order, offloading_order)
        return state

    def get_offloading(self, split_layer):
        offloading = {}
        workload = 0
        assert len(split_layer) == len(config.CLIENTS_LIST)
        for i in range(len(config.CLIENTS_LIST)):
            for l in range(len(config.model_cfg[config.model_name])):
                if l <= split_layer[i]:
                    workload += config.model_cfg[config.model_name][l][5]
            offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
            workload = 0
        return offloading

    def reinitialize(self, split_layers, offload, first, LR):
        self.initialize(split_layers, offload, first, LR)

    def scatter(self, msg):
        for ip, sock in self.client_socks.items():
            self.send_msg(sock, msg)
