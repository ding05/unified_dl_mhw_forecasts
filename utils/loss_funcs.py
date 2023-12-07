import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def bmc_loss(pred, target, noise_var):
    """
    Compute the Balanced MSE Loss (BMC) between 'pred' and the ground truth 'target'.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    #logits = - (pred - target.T).pow(2) / (2 * noise_var) # Logit size: [batch, batch], deprecated
    logits = - (pred - target.permute(*torch.arange(target.ndim - 1, -1, -1))).pow(2) / (2 * noise_var)
    #logits = - (pred - target.unsqueeze(1)).pow(2) / (2 * noise_var) # An alternative, it may return a type error.
    #print('noise_var:', float(noise_var.detach()))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).float()) # Contrastive-like loss
    loss = loss * (2 * noise_var).detach() # Optional: restore the loss scale, 'detach' when noise is learnable.
    return loss, noise_var

class BMCLoss(torch.nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

def cm_weighted_mse(preds, targets, threshold, alpha=1.5, beta=0.5, weight=1.0):
    # Ensure the threshold tensor has the same dimensions as targets.
    threshold = threshold.view_as(targets)
    # Calculate the weights using a tensor mask for conditional computation.
    mask = targets > threshold
    weights = ((weight * torch.abs(targets) ** alpha) * mask + (torch.abs(targets) ** alpha) * (~mask)) ** beta
    unweighted_loss = (preds - targets) ** 2
    loss = unweighted_loss * weights
    loss = loss.mean()
    return loss