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

# Calculate the symmetric extremal dependence index (SEDI) node by node.
def calculate_sedi(pred_feats, test_feats, threshold):
    # Hit rate (recall)
    H = calculate_recall(pred_feats, test_feats, threshold)
    pred_pos = pred_feats > threshold
    true_pos = test_feats > threshold
    true_neg = ~(test_feats > threshold)
    TP = np.logical_and(pred_pos, true_pos).sum()
    FP = np.logical_and(pred_pos, ~true_pos).sum()
    TN = np.logical_and(~pred_pos, true_neg).sum()
    # Avoid division by zero for False Alarm Rate.
    denominator = FP + TN
    if denominator == 0:
        F = np.nan
    else:
        F = FP / denominator
    # SEDI alculation
    logF = np.log(F) if F != 0 else np.nan
    logH = np.log(H) if H != 0 else np.nan
    log1F = np.log(1 - F) if F != 1 else np.nan
    log1H = np.log(1 - H) if H != 1 else np.nan
    # If any of the values is nan (due to log(0) or log(1)), then SEDI should be nan.
    if np.isnan(logF) or np.isnan(logH) or np.isnan(log1F) or np.isnan(log1H):
        return np.nan
    else:
        return (logF - logH - log1F + log1H) / (logF + logH + log1F + log1H)