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

# Add an LSTM layer to process node-level time series.
class MultiGraphSage_LSTM(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_graphs, aggr='mean'):
        super(MultiGraphSage_LSTM, self).__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Sequential(SAGEConv(in_channels, hid_channels, aggr=aggr), SAGEConv(hid_channels, out_channels, aggr=aggr)) for _ in range(num_graphs)])
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=out_channels, batch_first=True)

    def forward(self, data_sequence):
        x_seq_list = []
        for data_list in data_sequence:
            x_list = []
            for i, data in enumerate(data_list):
                x = data.x
                for j, layer in enumerate(self.convs[i]):
                    x = layer(x, data.edge_index)
                    x = torch.tanh(x)
                x_list.append(x)
            # Transpose to get sequences for each node.
            x_seq = torch.stack(x_list).transpose(0, 1) # x_seq's shape: (num_nodes, sequence_length, feature_size)
            x_seq_list.append(x_seq)
        # Combine sequences for the batch.    
        lstm_input = torch.cat(x_seq_list, dim=0)
        lstm_out, _ = self.lstm(lstm_input)
        # Only return the last timestep's prediction for each sequence.
        return lstm_out[:, -1, :].squeeze()