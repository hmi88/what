import os
import math
import sys
from datetime import datetime
from functools import reduce

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.nn.modules.module import _addindent


class Checkpoint:
    def __init__(self, config):
        self.global_step = 0
        self.last_epoch = 0
        self.config = config
        self.exp_dir = config.exp_dir
        self.exp_load = config.exp_load
        exp_type = config.uncertainty
        now = datetime.now().strftime('%m%d_%H%M')

        if config.exp_load is None:
            dir_fmt = '{}/{}_{}'.format(config.data_name, exp_type, now)
        else:
            dir_fmt = '{}/{}_{}'.format(config.data_name, exp_type, self.exp_load)

        self.model_dir = os.path.join(self.exp_dir, dir_fmt, 'model')
        self.log_dir = os.path.join(self.exp_dir, dir_fmt, 'log')
        self.save_dir = os.path.join(self.exp_dir, dir_fmt, 'save')
        self.ckpt_dir = os.path.join(self.log_dir, 'ckpt.pt')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # save config
        self.config_file = os.path.join(self.log_dir, 'config.txt')
        with open(self.config_file, 'w') as f:
            for k, v in vars(config).items():
                f.writelines('{}: {} \n'.format(k, v))

    def step(self):
        self.global_step += 1
        return self.global_step

    def save(self, epoch):
        self.last_epoch = epoch
        save_ckpt = {'global_step': self.global_step,
                     'last_epoch': self.last_epoch}
        torch.save(save_ckpt, self.ckpt_dir)

    def load(self):
        load_ckpt = torch.load(self.ckpt_dir)
        self.global_step = load_ckpt['global_step']
        self.last_epoch = load_ckpt['last_epoch']


def calc_psnr(output, label, rgb_range=1.):
    if label.nelement() == 1: return 0

    diff = (output - label) / rgb_range
    mse = diff.pow(2).mean()

    return -10 * math.log10(mse)


def make_optimizer(config, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    kwargs_optimizer = {'lr': config.lr, 'weight_decay': config.weight_decay}

    if config.optimizer == 'sgd':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = config.momentum
    elif config.optimizer == 'adam':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = config.betas
        kwargs_optimizer['eps'] = config.epsilon
    elif config.optimizer == 'rmsprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = config.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), config.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': config.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, ckpt):
            save_dir = os.path.join(ckpt.model_dir, 'optimizer.pt')
            torch.save(self.state_dict(), save_dir)

        def load(self, ckpt):
            load_dir = os.path.join(ckpt.model_dir, 'optimizer.pt')
            epoch = ckpt.last_epoch
            self.load_state_dict(torch.load(load_dir))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


def summary(model, config_file, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    print(string, file=open(config_file, 'a'))

    if file is not None:
        print(string, file=sys.stdout)
        file.flush()

    return count

