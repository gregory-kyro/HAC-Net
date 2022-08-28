import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter as Param
from torch_sparse import coalesce
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool, NNConv, avg_pool_x
from torch_geometric.nn.inits import uniform, reset
from torch_geometric.utils import add_self_loops, is_undirected, to_undirected, contains_self_loops
from torch_geometric.nn.aggr import AttentionalAggregation


''' Define Graph Cutoff class '''
class Graph_Cutoff(torch.nn.Module):

    def __init__(self, t):
        super(Graph_Cutoff, self).__init__()
        if torch.cuda.is_available():
            self.t = nn.Parameter(t, requires_grad=True).cuda()
        else:
            self.t = nn.Parameter(t, requires_grad=True)

    def filter_adj(self, row, col, edge_feat, mask):
        mask = mask.squeeze()
        return row[mask], col[mask], None if edge_feat is None else edge_feat[mask]

    def forward(self, edge_index, edge_feat):
        num = maybe_num_nodes(edge_index, None)
        row, col = edge_index
        mask = edge_feat <= self.t
        row, col, edge_feat = self.filter_adj(row, col, edge_feat, mask)
        edge_index = torch.stack([torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0)
        edge_feat = torch.cat([edge_feat, edge_feat], dim=0)
        edge_index, edge_feat = coalesce(edge_index, edge_feat, num, num)
        return edge_index, edge_feat


''' The gated graph operator from 'Gated Graph Sequence Neural Networks':
<https://arxiv.org/abs/1511.05493> '''
class Gated_Graph(MessagePassing):

    def __init__(self, out_channels, num_layers, edge_network, aggregation, bias=True):
        super(Gated_Graph, self).__init__(aggregation)
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.edge_network = edge_network
        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.RNN = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()
        self.aggregation=aggregation

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.RNN.reset_parameters()

    def forward(self, x, edge_index, edge_feat):
        node_feat = x if x.dim() == 2 else x.unsqueeze(-1)
        assert node_feat.size(1) <= self.out_channels
        # if input size <= out_channels, pad input with 0s to achieve the same size
        if node_feat.size(1) < self.out_channels:
            zero = node_feat.new_zeros(node_feat.size(0), self.out_channels - node_feat.size(1))
            node_fea = torch.cat([node_feat, zero], dim=1)
        for i in range(self.num_layers):
            mat_mul = torch.matmul(node_feat, self.weight[i])
            mat_mul = self.propagate(edge_index=edge_index, x=node_feat, aggregation=self.aggregation)
            node_feat = self.RNN(mat_mul, node_feat)
        return node_feat

    # define function to contruct message
    def message(self, x_j): 
        return x_j

    # define function to update node features
    def update(self, aggr_output):
        return aggr_output

    def __repr__(self):
        return "{}({}, num_layers={})".format(self.__class__.__name__, self.out_channels, self.num_layers)


''' Define Message Attention class '''
class Message_Attention(torch.nn.Module):

    # weight the message-passed features by Softmaxed original features
    def __init__(self, net_i, net_j):
        super(Message_Attention, self).__init__()
        self.net_i = net_i
        self.net_j = net_j

    def forward(self, feat_i, feat_j):
        return torch.nn.Softmax(dim=1)(self.net_i(torch.cat([feat_i, feat_j], dim=1))) * self.net_j(feat_j)


''' Define Propagation class '''
class Propagation(torch.nn.Module):

    def __init__(self, feat_size, gather_width, prop_iter, dist_cutoff):
        super(Propagation, self).__init__()
        assert dist_cutoff is not None
        self.dist_cutoff = dist_cutoff
        self.edge_feat_size = 1
        self.prop_iter = prop_iter
        self.gather_width = gather_width
        self.feat_size = feat_size
        self.gate_net = nn.Sequential(nn.Linear(self.feat_size, int(self.feat_size/2)), nn.Softsign(), nn.Linear(int(self.feat_size/2), int(self.feat_size/4)), nn.Softsign(), nn.Linear(int(self.feat_size/4),1))
        # Attentional aggregation
        self.attn_aggr = AttentionalAggregation(self.gate_net)
        self.edge_net = nn.Sequential(nn.Linear(self.edge_feat_size, int(self.feat_size / 2)), nn.Softsign(), nn.Linear(int(self.feat_size / 2), self.feat_size), nn.Softsign())
        self.edge_network = NNConv(self.feat_size, self.edge_feat_size * self.feat_size, nn=self.edge_net, root_weight=True, aggr="add")
        self.gate = Gated_Graph(self.feat_size, self.prop_iter, edge_network=self.edge_network, aggregation=self.attn_aggr)
        self.attention = Message_Attention(net_i=nn.Sequential(nn.Linear(self.feat_size * 2, self.feat_size), nn.Softsign(),nn.Linear(self.feat_size, self.gather_width), nn.Softsign()),
                                           net_j=nn.Sequential(nn.Linear(self.feat_size, self.gather_width), nn.Softsign()))
    # define forward propagation
    def forward(self, data, edge_index, edge_feat):
        node_feat_0 = data
        node_feat_1 = self.gate(node_feat_0, edge_index, edge_feat)
        node_feat_1 = self.attention(node_feat_1, node_feat_0)
        return node_feat_1


''' Define a Fully-Connected Network class '''
class Fully_Connected_Net(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Fully_Connected_Net, self).__init__()
        self.output = nn.Sequential(nn.Linear(in_channels, int(in_channels / 1.5)), nn.ReLU(), nn.Linear(int(in_channels / 1.5), int(in_channels / 2)), nn.ReLU(), nn.Linear(int(in_channels / 2), out_channels))

    def forward(self, data):
            return self.output(data)


''' Define full Message-Passing GCN class '''
class MP_GCN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, gather_width=128, prop_iter=4, dist_cutoff=3.5):
        super(MP_GCN, self).__init__()
        assert dist_cutoff is not None

        if torch.cuda.is_available():
            self.dist_cutoff = Graph_Cutoff(torch.ones(1).cuda() * dist_cutoff)
        else:
            self.dist_cutoff = Graph_Cutoff(torch.ones(1) * dist_cutoff)

        self.global_add_pool = global_add_pool
        self.propagation = Propagation(feat_size=in_channels, gather_width=gather_width, prop_iter=prop_iter, dist_cutoff=self.dist_cutoff)
        self.global_add_pool = global_add_pool
        self.output = Fully_Connected_Net(gather_width, out_channels)

    def forward(self, data):
        if torch.cuda.is_available():
            data.x = data.x.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.edge_index = data.edge_index.cuda()
            data.batch = data.batch.cuda()

        # make sure graph is undirected
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)

        # make sure that nodes can propagate messages to themselves
        if not contains_self_loops(data.edge_index):
            data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr.view(-1))

        # add self loops to enable self-propagation
        (edge_index, edge_feat) = self.dist_cutoff(data.edge_index, data.edge_attr)

        # propagation
        prop_x = self.propagation(data.x, edge_index, edge_feat)

        # gather features
        pool_x = self.global_add_pool(prop_x, data.batch)
        return self.output(pool_x)


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def filter_adj(row, col, edge_feat, mask):
    return row[mask], col[mask], None if edge_feat is None else edge_feat[mask]
