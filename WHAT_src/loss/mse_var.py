import torch
import torch.nn as nn
import torch.nn.functional as F


class MSE_VAR(nn.Module):
    def __init__(self):
        super(MSE_VAR, self).__init__()

    def forward(self, results, label):
        mean, var = results['mean'], results['var']

        loss1 = (torch.exp(-var) * ((mean - label) ** 2)).mean()
        loss2 = var.mean()

        loss = loss1 + loss2
        return loss

