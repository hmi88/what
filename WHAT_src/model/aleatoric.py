import torch.nn as nn

from model import common


def make_model(args):
    return ALEATORIC(args)


class ALEATORIC(nn.Module):
    def __init__(self, config):
        super(ALEATORIC, self).__init__()
        self.is_train = config.is_train
        in_channels = config.in_channels
        n_feats = config.n_feats

        # define head module
        head = [common.double_conv(in_channels, n_feats)]

        # define encoder module
        encoder = [common.down(n_feats*(i+1), n_feats*(i+2)) for i in range(2)]

        # define decoder_mean module
        decoder_mean = [common.up(n_feats*(i+2), n_feats*(i+1))
                          for i in reversed(range(2))]
        decoder_mean.append(common.default_conv(n_feats, in_channels, 'sigmoid'))

        # define decoder_var module
        decoder_var = [common.up(n_feats*(i+2), n_feats*(i+1))
                         for i in reversed(range(2))]
        decoder_var.append(common.default_conv(n_feats, in_channels))

        self.head = nn.Sequential(*head)
        self.encoder = nn.Sequential(*encoder)
        self.decoder_mean = nn.Sequential(*decoder_mean)
        self.decoder_var = nn.Sequential(*decoder_var)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x_head = self.head(x)
        x_enc = self.encoder(x_head)
        x_enc = self.dropout(x_enc)
        x_mean = self.decoder_mean(x_enc)
        x_var = self.decoder_var(x_enc)

        results = {'mean': x_mean, 'var': x_var}
        return results
