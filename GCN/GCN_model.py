import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.aggr import AttentionalAggregation

''' Define full Message-Passing GCN class '''

class MP_GCN(torch.nn.Module):

    def __init__(self, in_channels, gather_width=128, prop_iter=4, dist_cutoff=3.5):
        super(MP_GCN, self).__init__()

        #define distance cutoff
        self.dist_cutoff=torch.Tensor([dist_cutoff])
        if torch.cuda.is_available():
            self.dist_cutoff = self.dist_cutoff.cuda()

        #Attentional aggregation
        self.gate_net = nn.Sequential(nn.Linear(in_channels, int(in_channels/2)), nn.Softsign(), nn.Linear(int(in_channels/2), int(in_channels/4)), nn.Softsign(), nn.Linear(int(in_channels/4),1))
        self.attn_aggr = AttentionalAggregation(self.gate_net)
        
        #Gated Graph Neural Network
        self.gate = GatedGraphConv(in_channels, prop_iter, aggregation=self.attn_aggr)

        #Simple neural networks for use in asymmetric attentional aggregation
        self.attn_net_i=nn.Sequential(nn.Linear(in_channels * 2, in_channels), nn.Softsign(),nn.Linear(in_channels, gather_width), nn.Softsign())
        self.attn_net_j=nn.Sequential(nn.Linear(in_channels, gather_width), nn.Softsign())

        #Final set of linear layers for making affinity prediction
        self.output = nn.Sequential(nn.Linear(gather_width, int(gather_width / 1.5)), nn.ReLU(), nn.Linear(int(gather_width / 1.5), int(gather_width / 2)), nn.ReLU(), nn.Linear(int(gather_width / 2), 1))

    def forward(self, data):

        #Move data to GPU
        if torch.cuda.is_available():
            data.x = data.x.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.edge_index = data.edge_index.cuda()
            data.batch = data.batch.cuda()

        # allow nodes to propagate messages to themselves
        data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr.view(-1))

        # restrict edges to the distance cutoff
        row, col = data.edge_index
        mask = data.edge_attr <= self.dist_cutoff
        mask = mask.squeeze()
        row, col, edge_feat = row[mask], col[mask], data.edge_attr[mask]
        edge_index=torch.stack([row,col],dim=0)

        # propagation
        node_feat_0 = data.x
        node_feat_1 = self.gate(node_feat_0, edge_index, edge_feat)
        node_feat_attn = torch.nn.Softmax(dim=1)(self.attn_net_i(torch.cat([node_feat_1, node_feat_0], dim=1))) * self.attn_net_j(node_feat_0)

        # globally sum features and apply linear layers
        pool_x = global_add_pool(node_feat_attn, data.batch)
        prediction = self.output(pool_x)

        return prediction
