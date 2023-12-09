from utils.gnns import *
from utils.loss_funcs import *
from utils.process_utils import *
from utils.eval_utils import *

import numpy as np
from numpy import asarray, save, load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index

import time

data_path = 'data/'
models_path = 'configs/'
out_path = 'out/'


node_feat_filename = 'era5_node_feats_ssta_1980_2010.npy'
adj_filename = 'era5_adj_mat_0.8.npy'

window_size = 12
lead_time = 3
learning_rate = 0.01 # 0.001 for SSTs with MSE # 0.0005, 0.001 for RMSProp for SSTs
#learning_rate = 0.01 # For the GraphSAGE-LSTM
weight_decay = 0.0001 # 0.0001 for RMSProp
momentum = 0.9
l1_ratio = 1
num_epochs = 1 #1000, 400, 200
# Early stopping, if the validation MSE has not improved for "patience" epochs, stop training.
patience = num_epochs #100, 40, 20
min_val_mse = np.inf
# For the GraphSAGE-LSTM
sequence_length = 12

# Load the data.

node_feat_grid = load(data_path + node_feat_filename)
print('Node feature grid in Kelvin:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()

# Normalize the data to [0, 1].
node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid[:,:852])) / (np.max(node_feat_grid[:,:852]) - np.min(node_feat_grid[:,:852]))
print('Normalized node feature grid:', node_feat_grid_normalized)
print('Shape:', node_feat_grid_normalized.shape)
print('----------')
print()

adj_mat = load(data_path + adj_filename)
print('Adjacency matrix:', adj_mat)
print('Shape:', adj_mat.shape)
print('----------')
print()

# Compute the total number of time steps.
num_time = node_feat_grid.shape[1] - window_size - lead_time + 1

# Set the number of decimals in torch tensors printed.
torch.set_printoptions(precision=8)

# If lead time is greater than 1, use diffusion to interpolate the next time steps.
if lead_time > 1:
    
    # Generate PyG graphs for the interpolator network.
    graph_list = []
    for time_i in range(num_time):
        x = []
        y = []
        for node_i in range(node_feat_grid.shape[0]):
            x.append(node_feat_grid_normalized[node_i][time_i : time_i + window_size])
            x.append(0) # Initialize the first feature as zero, later replaced by the predicted feature at the target time.
            y.append(node_feat_grid_normalized[node_i][time_i + window_size + lead_time - 1])
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        # Generate incomplete graphs with the adjacency matrix.
        edge_index = torch.tensor(adj_mat, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
        graph_list.append(data)
    
    print('Inputs of the first node in the first graph, i.e. the first time step:', graph_list[0].x[0])
    print('Check if they match those in the node features:', node_feat_grid[0][:13])
    print('Check if they match those in the normalized node features:', node_feat_grid_normalized[0][:13])
    print('----------')
    print()
    
    # Split the data.
    
    train_graph_list = graph_list[:840]
    val_graph_list = graph_list[840:-1]
    test_graph_list = graph_list[840:-1]
    
    test_node_feats = node_feat_grid_normalized[:, 840 + window_size + lead_time - 1:-1]
    
    # Compute the percentiles using the training set only.
    node_feats_90 = np.percentile(node_feat_grid[:, :840], 90, axis=1)
    node_feats_95 = np.percentile(node_feat_grid[:, :840], 95, axis=1)
    node_feats_normalized_90 = np.percentile(node_feat_grid_normalized[:, :840], 90, axis=1)
    node_feats_normalized_95 = np.percentile(node_feat_grid_normalized[:, :840], 95, axis=1)
    
    # Select one threshold array.
    threshold_tensor = torch.tensor(node_feats_normalized_90).float()
    
    # Define the models.
    # Several interpolators and one forecaster
    interpolators = {}
    for i in range(1, lead_time):
        interpolators[i], model_class = MultiGraphSage_Dropout(in_channels=graph_list[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list), aggr='mean'), f'SAGE_ITP_{i}'
    forecaster, model_class = MultiGraphSage(in_channels=graph_list[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list), aggr='mean'), 'SAGE_FCS'
    
    # Define the loss function.
    criterion = nn.MSELoss()
    #criterion = BMCLoss(0.1)
    criterion_test = nn.MSELoss()
    
    # Define the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train a multi-graph GNN model.
    
    print('Start training.')
    print('----------')
    print()
    
    # Start time
    start = time.time()
    
    # Record the results by epoch.
    loss_epochs = []
    val_mse_nodes_epochs = []
    val_precision_nodes_epochs = []
    val_recall_nodes_epochs = []
    val_csi_nodes_epochs = []
    # Early stopping starting counter
    counter = 0

    for epoch in range(num_epochs):
        # Train the model.
        #model.train()
        # Iterate over the lead time to train the interpolators and train/refine the forecaster.
        for i in range(1, lead_time):
            
            # Iterate over the training data.
            # Train/refine the forecaster.
            for data in train_graph_list:
                optimizer.zero_grad()
                output = forecaster([data])
                loss = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
                #loss = cm_weighted_mse(output.squeeze(), torch.tensor(data.y).squeeze(), threshold=threshold_tensor)
                #loss = cm_weighted_mse(output.squeeze(), torch.tensor(data.y).squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
                loss.backward()
                optimizer.step()
            loss_epochs.append(loss.item())
            
            # Get the forecasted output and update the graphs.
            pred_node_feat_list.append(output.squeeze())
            for graph in train_graph_list:
                x = graph.x
                for node_i in range(x.shape[0]):
                    x[node_i, -1] = pred_node_feat_list[node_i]
                graph.x = x
            
            # Define the graph set for training an interpolator.
            graph_itp_list = []
            for time_i in range(num_time):
                x = []
                y = []
                for node_i in range(node_feat_grid.shape[0]):
                    x.append(node_feat_grid_normalized[node_i][time_i : time_i + window_size])
                    x.append(0) # Initialize the first feature as zero, later replaced by the predicted feature at the target time.
                    y.append(node_feat_grid_normalized[node_i][time_i + window_size + i - 1])
                x = torch.tensor(np.array(x))
                y = torch.tensor(np.array(y))
                # Generate incomplete graphs with the adjacency matrix.
                edge_index = torch.tensor(adj_mat, dtype=torch.long)
                data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
                graph_itp_list.append(data)
                
            train_graph_itp_list = graph_itp_list[:840]
            
            # Update the graph with the predicted values at the target time.
            for graph in train_graph_itp_list:
                x = graph.x
                for node_i in range(x.shape[0]):
                    x[node_i, -1] = pred_node_feat_list[node_i]
                graph.x = x
                    
            # Train an interpolator.
            for data in train_graph_itp_list:
                optimizer.zero_grad()
                output = interpolators[i]([data])
                loss = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
                #loss = cm_weighted_mse(output.squeeze(), torch.tensor(data.y).squeeze(), threshold=threshold_tensor)
                #loss = cm_weighted_mse(output.squeeze(), torch.tensor(data.y).squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
                loss.backward()
                optimizer.step()
            loss_epochs.append(loss.item())  
            
            # Get the interpolated output and update the graphs.
            pred_node_feat_list.append(output.squeeze())
            for graph in train_graph_list:
                x = graph.x
                for node_i in range(x.shape[0]):
                    x[node_i, i - 1] = pred_node_feat_list[node_i]
                graph.x = x
            
            """
            # Evaluate the model.
            model.eval()
            
            # Compute the MSE, precision, recall, and critical success index (CSI) on the validation set.
            with torch.no_grad():
                val_mse_nodes = 0
                pred_node_feat_list = []
                
                for data in val_graph_list:
                    output = model([data])
                    val_mse = criterion_test(output.squeeze(), torch.tensor(data.y).squeeze())
                    #print('Val predictions:', output.squeeze().tolist()[::300])
                    #print('Val observations:', torch.tensor(data.y).squeeze().tolist()[::300])
                    val_mse_nodes += val_mse
                    
                    # The model output graph by graph, but we are interested in time series at node by node.
                    # Transform the shapes.
                    pred_node_feat_list.append(output.squeeze())
            """

# If lead time is 1, there is no need to use diffusion.
elif lead_time == 1:

    # Generate PyG graphs from NumPy arrays.
    graph_list = []
    for time_i in range(num_time):
        x = []
        y = []
        for node_i in range(node_feat_grid.shape[0]):
            x.append(node_feat_grid_normalized[node_i][time_i : time_i + window_size])
            y.append(node_feat_grid_normalized[node_i][time_i + window_size + lead_time - 1])
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        # Generate incomplete graphs with the adjacency matrix.
        edge_index = torch.tensor(adj_mat, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
        graph_list.append(data)
    
    print('Inputs of the first node in the first graph, i.e. the first time step:', graph_list[0].x[0])
    print('Check if they match those in the node features:', node_feat_grid[0][:13])
    print('Check if they match those in the normalized node features:', node_feat_grid_normalized[0][:13])
    print('----------')
    print()
    
    # Split the data.
    
    train_graph_list = graph_list[:840]
    val_graph_list = graph_list[840:-1]
    test_graph_list = graph_list[840:-1]
    
    test_node_feats = node_feat_grid_normalized[:, 840 + window_size + lead_time - 1:-1]
    
    # Compute the percentiles using the training set only.
    node_feats_90 = np.percentile(node_feat_grid[:, :840], 90, axis=1)
    node_feats_95 = np.percentile(node_feat_grid[:, :840], 95, axis=1)
    node_feats_normalized_90 = np.percentile(node_feat_grid_normalized[:, :840], 90, axis=1)
    node_feats_normalized_95 = np.percentile(node_feat_grid_normalized[:, :840], 95, axis=1)
    
    # Select one threshold array.
    threshold_tensor = torch.tensor(node_feats_normalized_90).float()
    
    # Define the model.
    model, model_class = MultiGraphSage(in_channels=graph_list[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list), aggr='mean'), 'SAGE'
    
    # Define the loss function.
    #criterion = nn.MSELoss()
    #criterion = BMCLoss(0.1)
    criterion_test = nn.MSELoss()
    
    # Define the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train a multi-graph GNN model.
    
    print('Start training.')
    print('----------')
    print()
    
    # Start time
    start = time.time()
    
    # Record the results by epoch.
    loss_epochs = []
    val_mse_nodes_epochs = []
    val_precision_nodes_epochs = []
    val_recall_nodes_epochs = []
    val_csi_nodes_epochs = []
    # Early stopping starting counter
    counter = 0

    for epoch in range(num_epochs):
        # Iterate over the training data.
        for data in train_graph_list:
            optimizer.zero_grad()
            output = model([data])
            #loss = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
            #loss = cm_weighted_mse(output.squeeze(), torch.tensor(data.y).squeeze(), threshold=threshold_tensor)
            loss = cm_weighted_mse(output.squeeze(), torch.tensor(data.y).squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
            loss.backward()
            optimizer.step()
        loss_epochs.append(loss.item())
    
        # Compute the MSE, precision, recall, and critical success index (CSI) on the validation set.
        with torch.no_grad():
            val_mse_nodes = 0
            val_precision_nodes = 0
            val_recall_nodes = 0
            val_csi_nodes = 0
            pred_node_feat_list = []
            
            for data in val_graph_list:
                output = model([data])
                val_mse = criterion_test(output.squeeze(), torch.tensor(data.y).squeeze())
                print('Val predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
                print('Val observations:', [round(i, 4) for i in torch.tensor(data.y).squeeze().tolist()[::300]])
                val_mse_nodes += val_mse
                
                # The model output graph by graph, but we are interested in time series at node by node.
                # Transform the shapes.
                pred_node_feat_list.append(output.squeeze())
    
            val_mse_nodes /= len(val_graph_list)
            val_mse_nodes_epochs.append(val_mse_nodes.item())
            
            pred_node_feat_tensor = torch.stack([tensor for tensor in pred_node_feat_list], dim=1)
            pred_node_feats = pred_node_feat_tensor.numpy()
            gnn_mse = np.mean((pred_node_feats - test_node_feats) ** 2, axis=1)
            
            # Precision
            val_precision_nodes = np.nanmean([calculate_precision(pred_node_feats[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_precision_nodes_epochs.append(val_precision_nodes.item())
            # Recall
            val_recall_nodes = np.nanmean([calculate_recall(pred_node_feats[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_recall_nodes_epochs.append(val_recall_nodes.item())
            # CSI
            val_csi_nodes = np.nanmean([calculate_csi(pred_node_feats[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_csi_nodes_epochs.append(val_csi_nodes.item())
    
        print('----------')
        print()
    
        # Print the current epoch and validation MSE.
        print('Epoch [{}/{}], Loss: {:.6f}, Validation MSE (calculated by column / graph): {:.6f}'.format(epoch + 1, num_epochs, loss.item(), val_mse_nodes))
        print('MSEs by node:', gnn_mse)
        print('Validation MSE, precision, recall, and CSI (calculated by row / time series at nodes): {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(np.mean(gnn_mse), val_precision_nodes, val_recall_nodes, val_csi_nodes))
        print('Loss by epoch:', [float('{:.6f}'.format(loss)) for loss in (loss_epochs[-20:] if len(loss_epochs) > 20 else loss_epochs)]) # Print the last 20 elements if the list is too long.
        print('Validation MSE by epoch:', [float('{:.6f}'.format(val_mse)) for val_mse in (val_mse_nodes_epochs[-20:] if len(val_mse_nodes_epochs) > 20 else val_mse_nodes_epochs)]) # Same as above.
        print('Validation precision by epoch:', [float('{:.6f}'.format(val_precision)) for val_precision in (val_precision_nodes_epochs[-20:] if len(val_precision_nodes_epochs) > 20 else val_precision_nodes_epochs)])
        print('Validation recall by epoch:', [float('{:.6f}'.format(val_recall)) for val_recall in (val_recall_nodes_epochs[-20:] if len(val_recall_nodes_epochs) > 20 else val_recall_nodes_epochs)])
        print('Validation CSI by epoch:', [float('{:.6f}'.format(val_csi)) for val_csi in (val_csi_nodes_epochs[-20:] if len(val_csi_nodes_epochs) > 20 else val_csi_nodes_epochs)])
        print('Persistence MSE:', ((test_node_feats[:,1:] - test_node_feats[:,:-1])**2).mean())
    
        # Update the best model weights if the current validation MSE is lower than the previous minimum.
        if val_mse_nodes.item() < min_val_mse:
            min_val_mse = val_mse_nodes.item()
            best_epoch = epoch
            best_model_weights = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_loss = loss
            best_pred_node_feats = pred_node_feats
            counter = 0
        else:
            counter += 1
        # If the validation MSE has not improved for "patience" epochs, stop training.
        if counter >= patience:
            print(f'Early stopping at Epoch {epoch} with best validation MSE: {min_val_mse} at Epoch {best_epoch}.')
            break
    
    print('----------')
    print()
    
    # End time
    stop = time.time()
    
    print(f'Complete training. Time spent: {stop - start} seconds.')
    print('----------')
    print()
    
    """
    # Test the model.
    with torch.no_grad():
        test_mse_nodes = 0
        for data in test_graph_list:
            output = model([data])
            test_mse = criterion_test(output.squeeze(), torch.tensor(data.y).squeeze())
            print('Test predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
            print('Test observations:', [round(i, 4) for i in torch.tensor(data.y).squeeze().tolist()[::300]])
            test_mse_nodes += test_mse
        test_mse_nodes /= len(test_graph_list)
        print('Test MSE: {:.4f}'.format(test_mse_nodes))
    
    print('----------')
    print()
    """
    
    # Save the results.
    save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_losses' + '.npy', np.array(loss_epochs))
    save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valmses' + '.npy', np.array(val_mse_nodes_epochs))
    save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valprecisions' + '.npy', np.array(val_precision_nodes_epochs))
    save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valrecalls' + '.npy', np.array(val_recall_nodes_epochs))
    save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valcsis' + '.npy', np.array(val_csi_nodes_epochs))
    save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_preds' + '.npy', best_pred_node_feats)
    save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_testobs' + '.npy', test_node_feats)
    
    print('Save the results in NPY files.')
    print('----------')
    print()
    
    # Save the model.
    torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': best_optimizer_state,
                'loss': best_loss
                }, models_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop))
    
    print('Save the checkpoint in a TAR file.')
    print('----------')
    print()

# Otherwise, print an error message.
else:
    print('Lead time error')