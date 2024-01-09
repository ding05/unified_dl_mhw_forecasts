import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class MultiGraphSage(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_graphs, aggr='mean'):
        super(MultiGraphSage, self).__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Sequential(SAGEConv(in_channels, hid_channels, aggr=aggr), SAGEConv(hid_channels, out_channels, aggr=aggr)) for _ in range(num_graphs)])
        
    def forward(self, data_list):
        x_list = []
        for i, data in enumerate(data_list):
            x = data.x
            for j, layer in enumerate(self.convs[i]):
                x = layer(x, data.edge_index)
                x = torch.tanh(x)
            x_list.append(x)
        x_concat = torch.cat(x_list, dim=0)
        return x_concat
        
class MultiGraphSage_Dropout(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_graphs, aggr='mean', dropout_rate=0.2):
        super(MultiGraphSage_Dropout, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                SAGEConv(in_channels, hid_channels, aggr=aggr),
                nn.Dropout(dropout_rate),
                SAGEConv(hid_channels, out_channels, aggr=aggr),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_graphs)
        ])

    def forward(self, data_list):
        x_list = []
        for i, data in enumerate(data_list):
            x = data.x
            for j, layer in enumerate(self.convs[i]):
                x = layer(x, data.edge_index) if j % 2 == 0 else layer(x) # Apply edge_index to SAGEConv and not to dropout.
                if j % 2 == 0:
                    x = torch.tanh(x)
            x_list.append(x)
        x_concat = torch.cat(x_list, dim=0)
        return x_concat