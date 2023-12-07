import numpy as np

# Calculate precision node by node.
def calculate_precision(pred_feats, test_feats, threshold):
    pred_pos = pred_feats > threshold
    true_pos = test_feats > threshold
    TP = np.logical_and(pred_pos, true_pos).sum()
    FP = np.logical_and(pred_pos, ~true_pos).sum()
    # Avoid division by zero.
    denominator = TP + FP
    if denominator == 0:
        return np.nan
    else:
        return TP / denominator

# Calculate recall node by node.
def calculate_recall(pred_feats, test_feats, threshold):
    pred_pos = pred_feats > threshold
    true_pos = test_feats > threshold
    TP = np.logical_and(pred_pos, true_pos).sum()
    FN = np.logical_and(~pred_pos, true_pos).sum()
    # Avoid division by zero.
    denominator = TP + FN
    if denominator == 0:
        return np.nan
    else:
        return TP / denominator

# Calculate the critical success index (CSI) node by node.
def calculate_csi(pred_feats, test_feats, threshold):
    pred_pos = pred_feats > threshold
    true_pos = test_feats > threshold
    TP = np.logical_and(pred_pos, true_pos).sum()
    FP = np.logical_and(pred_pos, ~true_pos).sum()
    FN = np.logical_and(~pred_pos, true_pos).sum()
    # Avoid division by zero.
    denominator = TP + FP + FN
    if denominator == 0:
        return np.nan
    else:
        return TP / denominator