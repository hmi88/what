import torch
import torch.nn as nn
import torch.nn.functional as F


class MSE_VAR(nn.Module):
    def __init__(self, var_weight):
        super(MSE_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, var = results['mean'], results['var']
        var = self.var_weight * var

        loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
        loss2 = var
        loss = .5 * (loss1 + loss2)
        return loss.mean()

