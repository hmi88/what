import torch.nn as nn
import torch.nn.functional as F

from model import common


def make_model(args):
    return NORMAL(args)


class NORMAL(nn.Module):
    def __init__(self, config):
        super(NORMAL, self).__init__()
        in_channels = config.in_channels
        n_feats = config.n_feats

        # define head module
        head = [common.double_conv(in_channels, n_feats)]

        # define encoder module
        encoder = [common.down(n_feats*(i+1), n_feats*(i+2)) for i in range(2)]

        # define decoder module
        decoder = [common.up(n_feats*(i+2), n_feats*(i+1))
                          for i in reversed(range(2))]
        decoder.append(common.default_conv(n_feats, in_channels, 'sigmoid'))

        self.head = nn.Sequential(*head)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_head = self.head(x)
        x_enc = self.encoder(x_head)
        x_enc = self.dropout(x_enc)
        x_mean = self.decoder(x_enc)

        results = {'mean': x_mean}
        return results

