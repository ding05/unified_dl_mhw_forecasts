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

node_feat_filename = 'node_feats_ssta_1980_2010.npy'
adj_filename = 'adj_mat_25.npy'

window_size = 12
lead_time = 1
loss_func = 'MSE' #'MSE', 'BMSE', 'WMSE'
learning_rate = 0.01 # 0.001 for SSTs with MSE # 0.0005, 0.001 for RMSProp for SSTs
#learning_rate = 0.01 # For the GraphSAGE-LSTM
weight_decay = 0.0001 # 0.0001 for RMSProp
#momentum = 0.9
l1_ratio = 1
num_epochs = 50 #1000, 400, 200
# Early stopping, if the validation MSE has not improved for "patience" epochs, stop training.
patience = 20 #100, 40, 20
min_val_mse = np.inf
max_val_sedi = -np.inf
# For the GraphSAGE-LSTM
sequence_length = 12
# For the GraphSAGE-Diffusion
dropout_rate = 0.1

# Load the data.

node_feat_grid = load(data_path + node_feat_filename)
print('Node feature grid in Kelvin:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')

# Normalize the data to [0, 1].
node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid[:,:852])) / (np.max(node_feat_grid[:,:852]) - np.min(node_feat_grid[:,:852]))
print('Normalized node feature grid:', node_feat_grid_normalized)
print('Shape:', node_feat_grid_normalized.shape)
print('----------')

adj_mat = load(data_path + adj_filename)
print('Adjacency matrix:', adj_mat)
print('Shape:', adj_mat.shape)
print('----------')

# Compute the total number of time steps.
num_time = node_feat_grid.shape[1] - window_size - lead_time + 1

# Set the number of decimals in torch tensors printed.
torch.set_printoptions(precision=8)

##### ##### ##### ##### #####
##### ##### ##### ##### #####

# If lead time is greater than 1, use diffusion to interpolate the next time steps.
if lead_time > 1:
    
    # Generate PyG graphs for Forecaster.
    graph_list_fc = []
    for time_i in range(num_time):
        x = []
        y = []
        for node_i in range(node_feat_grid.shape[0]):
            x_slice = list(node_feat_grid_normalized[node_i][time_i : time_i + window_size]) + [0] * (lead_time - 1) # Initialize the last feature(s) as zero, later replaced by the interpolated features before the target time.
            #print(x_slice)
            x.append(x_slice)
            y.append(node_feat_grid_normalized[node_i][time_i + window_size + lead_time - 1])
        x = torch.tensor(np.array(x), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.float)
        # Generate incomplete graphs with the adjacency matrix.
        edge_index = torch.tensor(adj_mat, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
        graph_list_fc.append(data)
    
    print('Inputs of the first node in the first graph, i.e. the first time step:', graph_list_fc[0].x[0])
    print('Check if they match those in the node features:', node_feat_grid[0][:13])
    print('Check if they match those in the normalized node features:', node_feat_grid_normalized[0][:13])
    print('----------')

    # Generate PyG graphs for Interpolator(s).
    graph_list_ipt = []
    for time_i in range(num_time):
        x = []
        y = []
        for node_i in range(node_feat_grid.shape[0]):
            x_slice = list(node_feat_grid_normalized[node_i][time_i : time_i + window_size]) + [0]  # Initialize the last feature as zero, later replaced by the forecasted feature at the target time.
            #print(x_slice)
            x.append(x_slice)
            y.append(node_feat_grid_normalized[node_i][time_i + window_size : time_i + window_size + lead_time - 1])
        x = torch.tensor(np.array(x), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.float)
        # Generate incomplete graphs with the adjacency matrix.
        edge_index = torch.tensor(adj_mat, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
        graph_list_ipt.append(data)
    
    print('Inputs of the first node in the first graph, i.e. the first time step:', graph_list_ipt[0].x[0])
    print('Check if they match those in the node features:', node_feat_grid[0][:13])
    print('Check if they match those in the normalized node features:', node_feat_grid_normalized[0][:13])
    print('----------')
    
    # Split the data.
    
    train_graph_list_fc = graph_list_fc[:840]
    val_graph_list_fc = graph_list_fc[840:-1]
    test_graph_list_fc = graph_list_fc[840:-1]

    train_graph_list_ipt = graph_list_ipt[:840]
    val_graph_list_ipt = graph_list_ipt[840:-1]
    test_graph_list_ipt = graph_list_ipt[840:-1]
    
    test_node_feats_fc = node_feat_grid_normalized[:, 840 + window_size + lead_time - 1:-1]
    
    # Compute the percentiles using the training set only.
    node_feats_90 = np.percentile(node_feat_grid[:, :840], 90, axis=1)
    node_feats_95 = np.percentile(node_feat_grid[:, :840], 95, axis=1)
    node_feats_normalized_90 = np.percentile(node_feat_grid_normalized[:, :840], 90, axis=1)
    node_feats_normalized_95 = np.percentile(node_feat_grid_normalized[:, :840], 95, axis=1)
    
    # Select one threshold array.
    threshold_tensor = torch.tensor(node_feats_normalized_90).float()
    
    # Define the models.
    # One or more Interpolators and one Forecaster
    model_class = 'SAGE_Diffus'
    interpolators = {}
    model_classes_ipt = {}
    for i in range(1, lead_time):
        interpolators[i], model_classes_ipt[i] = MultiGraphSage_Dropout(in_channels=graph_list_ipt[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list_ipt), aggr='mean', dropout_rate=dropout_rate), f'SAGE_ITP_{i}'
    forecaster, model_class_fc = MultiGraphSage(in_channels=graph_list_fc[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list_fc), aggr='mean'), 'SAGE_FC'
    
    # Define the loss function.
    if loss_func == 'MSE' or 'WMSE':
        criterion = nn.MSELoss()
    elif loss_func == 'BMSE':
        criterion = BMCLoss(0.02)
    else:
        print('Loss function error')
    criterion_test = nn.MSELoss()
    
    # Define the optimizer.
    optimizers_interpolator = {}
    for i in range(1, lead_time):
        optimizers_interpolator[i] = torch.optim.AdamW(interpolators[i].parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_forecaster = torch.optim.AdamW(forecaster.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train multi-graph GNN models.
    print('Start training.')
    print('----------')
    
    # Start time
    start = time.time()
    
    # Record the results by epoch.
    loss_epochs = []
    loss_epochs_fc = []
    loss_epochs_ipt = []
    val_mse_nodes_epochs = []
    val_precision_nodes_epochs = []
    val_recall_nodes_epochs = []
    val_csi_nodes_epochs = []
    val_sedi_nodes_epochs = []
    # Early stopping starting counter
    counter = 0

    for epoch in range(num_epochs):
        # Train the model.
        #model.train()
        # Iterate over the lead time to train Interpolator(s) and train/refine Forecaster.
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        pred_node_feat_list_fc = []
        
        for i in range(1, lead_time):
            
            # Iterate over the training data.
            print('Lead time [{}/{}]'.format(i, lead_time))
            pred_node_feat_list_ipt = []
            
            # Train/refine Forecaster.
            print('Train/refine Forecaster.')
            forecaster.train()
            for data in train_graph_list_fc:
                optimizer_forecaster.zero_grad()
                output = forecaster([data])
                if loss_func == 'MSE' or 'BMSE':
                    loss = criterion(output.squeeze(), data.y.squeeze())
                elif loss_func = 'WMSE':
                    #loss = cm_weighted_mse(output.squeeze(), data.y.squeeze(), threshold=threshold_tensor)
                    loss = cm_weighted_mse(output.squeeze(), data.y.squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
                else:
                    print('Loss function error')
                loss.backward()
                optimizer_forecaster.step()
                
                # Detach the tensor to prevent unwanted graph connections, which makes sure the output is not part of the computation graph in the next iteration.
                pred_node_feat_fc = output.squeeze().detach()
                pred_node_feat_list_fc.append(pred_node_feat_fc)
                
            loss_epochs_fc.append(loss.item())
                    
            # Get the forecasted output and update the graphs for training Interpolator(s).
            # Update graphs for training Interpolator(s).
            #print('Update graphs for training Interpolator(s).')
            for g in range(len(train_graph_list_ipt)):
                x = train_graph_list_ipt[g].x
                # Use cloned tensor to avoid in-place operation, which avoids modifying tensor in-place which could cause computation graph issues.
                for node_i in range(x.shape[0]):
                    x[node_i, -1] = pred_node_feat_list_fc[g][node_i].clone()
                train_graph_list_ipt[g].x = x
                    
            # Train one Interpolator.
            print('Train Interpolator [{}/{}].'.format(i, lead_time - 1))
            interpolators[i].train()
            for data in train_graph_list_ipt:
                optimizers_interpolator[i].zero_grad()
                output = interpolators[i]([data])
                if loss_func == 'MSE' or 'BMSE':
                    loss_ipt = criterion(output.squeeze(), data.y[:, i - 1].squeeze())
                elif loss_func == 'WMSE':
                    #loss_ipt = cm_weighted_mse(output.squeeze(), data.y[:, i - 1].squeeze(), threshold=threshold_tensor)
                    loss_ipt = cm_weighted_mse(output.squeeze(), data.y[:, i - 1].squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
                    #loss_ipt = cm_weighted_mse_2d(output.squeeze(), data.y[:, i - 1].squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
                else:
                    print('Loss function error')
                loss_ipt.backward()
                optimizers_interpolator[i].step()
                
                pred_node_feat_ipt = output.squeeze().detach()
                pred_node_feat_list_ipt.append(pred_node_feat_ipt)
            
            loss_epochs_ipt.append(loss_ipt.item())
            
            # Get the interpolated output and update the graphs for training Forecaster.
            # Update graphs for training Forecaster
            #print('Update graphs for training Forecaster.')
            for g in range(len(train_graph_list_fc)):
                x = train_graph_list_fc[g].x
                for node_i in range(x.shape[0]):
                    x[node_i, window_size + i - 1] = pred_node_feat_list_ipt[g][node_i].clone()
                train_graph_list_fc[g].x = x
            
        # Refine Forecaster for the last time for this epoch.
        print('Rrefine Forecaster.')
        for data in train_graph_list_fc:
            optimizer_forecaster.zero_grad()
            output = forecaster([data])
            if loss_func == 'MSE' or 'BMSE':
                loss = criterion(output.squeeze(), data.y.squeeze())
            elif loss_func == 'WMSE':
                #loss = cm_weighted_mse(output.squeeze(), data.y.squeeze(), threshold=threshold_tensor)
                loss = cm_weighted_mse(output.squeeze(), data.y.squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
            else:
                print('Loss function error')
            loss.backward()
            optimizer_forecaster.step()
            
        loss_epochs.append(loss.item()) # Save only the last loss value of Forecaster per epoch.
        
        # Evaluate the model.
        print('----------')
        print('Evaluate the models.')
        forecaster.eval()
        for i in range(1, lead_time):
             # Deactivate the dropout stochasticity of Interpolator(s).
             interpolators[i].eval()
             
             # Interpolate the values.
             with torch.no_grad():
                 val_mse_nodes = 0
                 pred_node_feat_list = []
                 
                 for data in val_graph_list_fc:
                     output = forecaster([data])
                     
                     pred_node_feat_fc = output.squeeze().detach()
                     pred_node_feat_list_fc.append(pred_node_feat_fc)
                 
                 for g in range(len(val_graph_list_ipt)):
                     x = val_graph_list_ipt[g].x
                     # Use cloned tensor to avoid in-place operation, which avoids modifying tensor in-place which could cause computation graph issues.
                     for node_i in range(x.shape[0]):
                         x[node_i, -1] = pred_node_feat_list_fc[g][node_i].clone()
                     val_graph_list_ipt[g].x = x

                 for data in val_graph_list_ipt:
                     output = interpolators[i]([data])
                    
                     pred_node_feat_ipt = output.squeeze().detach()
                     pred_node_feat_list_ipt.append(pred_node_feat_ipt)   
                
                 for g in range(len(val_graph_list_fc)):
                     x = val_graph_list_fc[g].x
                     for node_i in range(x.shape[0]):
                        x[node_i, window_size + i - 1] = pred_node_feat_list_ipt[g][node_i].clone()
                     val_graph_list_fc[g].x = x
                    
        # Evaluate Forecaster.
        # Compute the MSE, precision, recall, critical success index (CSI), and symmetric extremal dependence index (SEDI) on the validation set.
        with torch.no_grad():
            val_mse_nodes = 0
            val_precision_nodes = 0
            val_recall_nodes = 0
            val_csi_nodes = 0
            val_sedi_nodes = 0
            pred_node_feat_list = []
        
            for data in val_graph_list_fc:
                output = forecaster([data])
                val_mse = criterion_test(output.squeeze(), data.y.squeeze())
                #print('Val predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
                #print('Val observations:', [round(i, 4) for i in data.y.squeeze().tolist()[::300]])
                val_mse_nodes += val_mse
               
                pred_node_feat = output.squeeze().detach()
                pred_node_feat_list.append(pred_node_feat)
                
            val_mse_nodes /= len(val_graph_list_fc)
            val_mse_nodes_epochs.append(val_mse_nodes.item())
            
            pred_node_feat_tensor = torch.stack([tensor for tensor in pred_node_feat_list], dim=1)
            pred_node_feats = pred_node_feat_tensor.numpy()
            gnn_mse = np.mean((pred_node_feats - test_node_feats_fc) ** 2, axis=1)
            
            # Precision
            val_precision_nodes = np.nanmean([calculate_precision(pred_node_feats[i], test_node_feats_fc[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_precision_nodes_epochs.append(val_precision_nodes.item())
            # Recall
            val_recall_nodes = np.nanmean([calculate_recall(pred_node_feats[i], test_node_feats_fc[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_recall_nodes_epochs.append(val_recall_nodes.item())
            # CSI
            val_csi_nodes = np.nanmean([calculate_csi(pred_node_feats[i], test_node_feats_fc[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_csi_nodes_epochs.append(val_csi_nodes.item())
            # SEDI
            val_sedi_nodes = np.nanmean([calculate_sedi(pred_node_feats[i], test_node_feats_fc[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_sedi_nodes_epochs.append(val_sedi_nodes.item())

        # Print the current epoch and validation MSE.
        #print('Epoch [{}/{}], Loss: {:.6f}, Validation MSE (calculated by column / graph): {:.6f}'.format(epoch + 1, num_epochs, loss.item(), val_mse_nodes))
        print('MSEs by node:', gnn_mse)
        print('Validation MSE, precision, recall, CSI, and SEDI (calculated by row / time series at nodes): {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(np.mean(gnn_mse), val_precision_nodes, val_recall_nodes, val_csi_nodes, val_sedi_nodes))
        print('Loss by epoch:', [float('{:.6f}'.format(loss)) for loss in (loss_epochs[-20:] if len(loss_epochs) > 20 else loss_epochs)]) # Print the last 20 elements if the list is too long.
        print('Validation MSE by epoch:', [float('{:.6f}'.format(val_mse)) for val_mse in (val_mse_nodes_epochs[-20:] if len(val_mse_nodes_epochs) > 20 else val_mse_nodes_epochs)]) # Same as above.
        print('Validation precision by epoch:', [float('{:.6f}'.format(val_precision)) for val_precision in (val_precision_nodes_epochs[-20:] if len(val_precision_nodes_epochs) > 20 else val_precision_nodes_epochs)])
        print('Validation recall by epoch:', [float('{:.6f}'.format(val_recall)) for val_recall in (val_recall_nodes_epochs[-20:] if len(val_recall_nodes_epochs) > 20 else val_recall_nodes_epochs)])
        print('Validation CSI by epoch:', [float('{:.6f}'.format(val_csi)) for val_csi in (val_csi_nodes_epochs[-20:] if len(val_csi_nodes_epochs) > 20 else val_csi_nodes_epochs)])
        print('Validation SEDI by epoch:', [float('{:.6f}'.format(val_sedi)) for val_sedi in (val_sedi_nodes_epochs[-20:] if len(val_sedi_nodes_epochs) > 20 else val_sedi_nodes_epochs)])
        #print('Persistence MSE:', ((test_node_feats_fc[:,1:] - test_node_feats_fc[:,:-1])**2).mean())            

        # Current time
        cur = time.time()
        print(f'Time spent: {cur - start} seconds')
        print('----------')

        # Update the best model weights if the current validation SEDI is higher than the previous maximum.
        if val_sedi_nodes.item() > max_val_sedi:
            max_val_sedi = val_sedi_nodes.item()
            best_epoch = epoch
            best_interpolators_weights = {}
            best_optimizers_states_interpolators = {}
            for i in range(1, lead_time):
                best_interpolators_weights[i] = interpolators[i].state_dict()
                best_optimizers_states_interpolators[i] = optimizers_interpolator[i].state_dict()
            best_forecaster_weights = forecaster.state_dict()
            best_optimizer_state_forecaster = optimizer_forecaster.state_dict()
            best_loss = loss
            best_pred_node_feats = pred_node_feats
            counter = 0
        else:
            counter += 1
        # If the validation MSE has not improved for "patience" epochs, stop training.
        if counter >= patience:
            print(f'Early stopping at Epoch {epoch} with best validation SEDI: {min_sedi_mse} at Epoch {best_epoch}.')
            break

    # End time
    stop = time.time()

    # Save the results.
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_losses' + '.npy', np.array(loss_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_losses_fc' + '.npy', np.array(loss_epochs_fc))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valmses' + '.npy', np.array(val_mse_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valprecisions' + '.npy', np.array(val_precision_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valrecalls' + '.npy', np.array(val_recall_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valcsis' + '.npy', np.array(val_csi_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valsedis' + '.npy', np.array(val_sedi_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_preds' + '.npy', best_pred_node_feats)
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_testobs' + '.npy', test_node_feats_fc)

    print('Save the results in NPY files.')
    print('----------')

    # Save the models.
    # Save Interpolator(s).
    for i in range(1, lead_time):
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': best_interpolators_weights[i],
            'optimizer_state_dict': best_optimizers_states_interpolators[i],
            #'loss': best_loss
            }, models_path + model_class + '_' + loss_funces_ipt[i] + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop))
    # Save Forecaster.
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': best_forecaster_weights,
        'optimizer_state_dict': best_optimizer_state_forecaster,
        'loss': best_loss
        }, models_path + model_class + '_' + loss_func_fc + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop))
    
    print('Save the checkpoints in TAR files.')
    print('----------')

##### ##### ##### ##### #####
##### ##### ##### ##### #####
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
    if loss_func == 'MSE' or 'WMSE':
        criterion = nn.MSELoss()
    elif loss_func == 'BMSE':
        criterion = BMCLoss(0.02)
    else:
        print('Loss function error')
    criterion_test = nn.MSELoss()
    
    # Define the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train a multi-graph GNN model.
    
    print('Start training.')
    print('----------')
    
    # Start time
    start = time.time()
    
    # Record the results by epoch.
    loss_epochs = []
    val_mse_nodes_epochs = []
    val_precision_nodes_epochs = []
    val_recall_nodes_epochs = []
    val_csi_nodes_epochs = []
    val_sedi_nodes_epochs = []
    # Early stopping starting counter
    counter = 0

    for epoch in range(num_epochs):
        # Iterate over the training data.
        for data in train_graph_list:
            optimizer.zero_grad()
            output = model([data])
            if loss_func == 'MSE' or 'BMSE':
                loss = criterion(output.squeeze(), data.y.squeeze())
            elif: loss_func = 'WMSE':
                #loss = cm_weighted_mse(output.squeeze(), data.y.squeeze(), threshold=threshold_tensor)
                loss = cm_weighted_mse(output.squeeze(), data.y.squeeze(), threshold=threshold_tensor, alpha=2.0, beta=1.0, weight=2.0)
            else:
                print('Loss function error')
            loss.backward()
            optimizer.step()
        loss_epochs.append(loss.item())
    
        # Compute the MSE, precision, recall, critical success index (CSI), and symmetric extremal dependence index (SEDI) on the validation set.
        with torch.no_grad():
            val_mse_nodes = 0
            val_precision_nodes = 0
            val_recall_nodes = 0
            val_csi_nodes = 0
            val_sedi_nodes = 0
            pred_node_feat_list = []
            
            for data in val_graph_list:
                output = model([data])
                val_mse = criterion_test(output.squeeze(), data.y.squeeze())
                print('Val predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
                print('Val observations:', [round(i, 4) for i in data.y.squeeze().tolist()[::300]])
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
            # SEDI
            val_sedi_nodes = np.nanmean([calculate_sedi(pred_node_feats[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_sedi_nodes_epochs.append(val_sedi_nodes.item())
    
        print('----------')
    
        # Print the current epoch and validation MSE.
        print('Epoch [{}/{}], Loss: {:.6f}, Validation MSE (calculated by column / graph): {:.6f}'.format(epoch + 1, num_epochs, loss.item(), val_mse_nodes))
        print('MSEs by node:', gnn_mse)
        print('Validation MSE, precision, recall, CSI, and SEDI (calculated by row / time series at nodes): {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(np.mean(gnn_mse), val_precision_nodes, val_recall_nodes, val_csi_nodes, val_sedi_nodes))
        print('Loss by epoch:', [float('{:.6f}'.format(loss)) for loss in (loss_epochs[-20:] if len(loss_epochs) > 20 else loss_epochs)]) # Print the last 20 elements if the list is too long.
        print('Validation MSE by epoch:', [float('{:.6f}'.format(val_mse)) for val_mse in (val_mse_nodes_epochs[-20:] if len(val_mse_nodes_epochs) > 20 else val_mse_nodes_epochs)]) # Same as above.
        print('Validation precision by epoch:', [float('{:.6f}'.format(val_precision)) for val_precision in (val_precision_nodes_epochs[-20:] if len(val_precision_nodes_epochs) > 20 else val_precision_nodes_epochs)])
        print('Validation recall by epoch:', [float('{:.6f}'.format(val_recall)) for val_recall in (val_recall_nodes_epochs[-20:] if len(val_recall_nodes_epochs) > 20 else val_recall_nodes_epochs)])
        print('Validation CSI by epoch:', [float('{:.6f}'.format(val_csi)) for val_csi in (val_csi_nodes_epochs[-20:] if len(val_csi_nodes_epochs) > 20 else val_csi_nodes_epochs)])
        print('Validation SEDI by epoch:', [float('{:.6f}'.format(val_sedi)) for val_sedi in (val_sedi_nodes_epochs[-20:] if len(val_sedi_nodes_epochs) > 20 else val_sedi_nodes_epochs)])
        print('Persistence MSE:', ((test_node_feats[:,1:] - test_node_feats[:,:-1])**2).mean())
    
        # Update the best model weights if the current validation MSE is lower than the previous minimum.
        if val_sedi_nodes.item() > max_val_sedi:
            max_val_sedi = val_sedi_nodes.item()
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
    
    # End time
    stop = time.time()
    
    print(f'Complete training. Time spent: {stop - start} seconds.')
    print('----------')
    
    # Save the results.
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_losses' + '.npy', np.array(loss_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valmses' + '.npy', np.array(val_mse_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valprecisions' + '.npy', np.array(val_precision_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valrecalls' + '.npy', np.array(val_recall_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valcsis' + '.npy', np.array(val_csi_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_valsedis' + '.npy', np.array(val_sedi_nodes_epochs))
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_preds' + '.npy', best_pred_node_feats)
    save(out_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop) +  '_testobs' + '.npy', test_node_feats)
    
    print('Save the results in NPY files.')
    print('----------')
    
    # Save the model.
    torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': best_optimizer_state,
                'loss': best_loss
                }, models_path + model_class + '_' + loss_func + '_' + adj_filename[8:-4] + '_' + str(lead_time) + '_' + str(stop))
    
    print('Save the checkpoint in a TAR file.')
    print('----------')

##### ##### ##### ##### #####
##### ##### ##### ##### #####
# Otherwise, print an error message.
else:
    print('Lead time error')