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
            if self.uncertainty=='aleatoric' or self.uncertainty=='normal':
                return forward_func(input)
            else:
                return self.forward_sample(input, forward_func=forward_func)

    def forward_sample(self, input, forward_func=None):
        output = []
        for i_sample in range(self.n_samples):
            output.append(forward_func(input)['mean'])

        mean = torch.stack(output, dim=0).mean(dim=0)
        var = torch.stack(output, dim=0).var(dim=0)
        results = {'mean': mean, 'var': var}
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