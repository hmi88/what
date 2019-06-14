import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        print('Making model...')

        self.is_train = config.is_train
        self.num_gpu = config.num_gpu
        self.uncertainty = config.uncertainty
        self.n_samples = config.n_samples
        module = import_module('model.' + config.uncertainty)
        self.model = module.make_model(config).to(config.device)

    def forward(self, input):
        if self.model.training:
            if self.num_gpu > 1:
                return P.data_parallel(self.model, input,
                                       list(range(self.num_gpu)))
            else:
                return self.model.forward(input)
        else:
            forward_func = self.model.forward
            if self.uncertainty == 'aleatoric':
                return self.test_aleatoric(input, forward_func)
            elif self.uncertainty == 'epistemic':
                return self.test_epistemic(input, forward_func)
            elif self.uncertainty == 'combined':
                return self.test_combined(input, forward_func)

    def test_aleatoric(self, input, forward_func):
        mean, var = forward_func(input)
        var = torch.exp(var)
        var_norm = var / var.max()
        results = {'mean': mean, 'var': var_norm}
        return results

    def test_epistemic(self, input, forward_func):
        mean1 = []
        mean2 = []
        for i_sample in range(self.n_samples):
            y = forward_func(input)['mean']
            mean1.append(y)
            mean2.append(y ** 2)
        mean1 = torch.stack(mean1, dim=0).mean(dim=0)
        mean2 = torch.stack((mean2), dim=0).mean(dim=0)
        var = mean1**2 - mean2
        results = {'mean': mean1, 'var': var}
        return results

    def test_combined(self, input, forward_func):
        mean1 = []
        mean2 = []
        var1 = []
        for i_sample in range(self.n_samples):
            y = forward_func(input)['mean']
            v = forward_func(input)['var']
            mean1.append(y ** 2)
            mean2.append(y)
            var1.append(torch.exp(v))

        mean1_ = torch.stack(mean1, dim=0).mean(dim=0)
        mean2_ = torch.stack(mean2, dim=0).mean(dim=0)
        var1_ = torch.stack(var1, dim=0).mean(dim=0)
        var = mean1_ - mean2_**2 + var1_
        results = {'mean': mean1, 'var': var}
        return results

    def save(self, ckpt, epoch):
        save_dirs = [os.path.join(ckpt.model_dir, 'model_latest.pt')]
        save_dirs.append(
            os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if epoch == -1:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
