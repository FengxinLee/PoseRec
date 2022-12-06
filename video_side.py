import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

'''
    This module is developed based on:
        https://github.com/fendou201398/st-gcn
'''

class Graph():
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)                             
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)    
        self.get_adjacency(strategy)                      

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        '''
            get pose graph
        '''
        if layout == 'blazepose':
            self.num_node = 33
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (10, 9), (11, 12), (12, 14),
                              (14, 16), (16, 22), (16, 18), (18, 20), (16, 20),
                              (11, 13), (13, 15), (15, 21), (15, 17), (15, 19),
                              (17, 19), (24, 12), (11, 23), (23, 24), (24, 26),
                              (26, 28), (28, 30), (28, 32), (30, 32), (23, 25),
                              (25, 27), (27, 29), (27, 31), (29, 31)]
            self.edge = self_link + neighbor_link
            self.center = 23    # (23 + 24)/2
        else:
            raise ValueError("Do Not Exist This Layout.")
        

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)    #[0,1,...,max_hop]
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]    # matrix_power 计算矩阵的指数
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class PoseModel_Disentangle(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, embedding_size, edge_importance_weighting, layer,num_cat):
        super().__init__()

        # load graph
        graph_args = {'layout': 'blazepose',
                      'strategy': 'uniform', 'dilation': 1, 'max_hop': 1}
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)
        self.num_cat = num_cat
        self.embedding_size = int(embedding_size//num_cat)

        # build networks
        self.in_channels = 4
        kernel_size = (9, A.size(0))
        self.data_bn = nn.BatchNorm1d(self.in_channels * A.size(1))

        if layer == 3:
            self.st_gcn_networks = nn.ModuleList((
                st_gcn(self.in_channels, 64, kernel_size, 1, residual=True, ),
                st_gcn(64, 128, kernel_size, 2, ),
                st_gcn(128, 256, kernel_size, 1, ),
            ))
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        self.fc = nn.Linear(256, embedding_size)

    def forward(self, x):
        # Tensor N×C×T×V×M
        # 0 N: batch size
        # 1 C: channel, default=4, (x,y,z,visibility)
        # 2 T: video frames
        # 3 V: pose landmarks, default=33
        # 4 M: number of people, default=1

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()    # N*M*V*C*T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()    # N*M*C*T*V
        x = x.view(N * M, C, T, V)

        for stgcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = stgcn(x, self.A*importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = []
        for i in range(self.num_cat):
            out.append(x[:, i*self.embedding_size:(i+1)*self.embedding_size])
        return out


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(
            in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels,
                      (kernel_size[0], 1), (stride, 1), padding,),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A

