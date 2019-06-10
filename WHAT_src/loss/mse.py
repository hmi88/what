import torch
import torch.nn as nn
import torch.nn.functional as F


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, results, label):
        mean = results['mean']
        loss = F.mse_loss(mean, label)
        return loss
