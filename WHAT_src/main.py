import torch

from config import get_config
from data import get_dataloader
from op import Operator
from util import Checkpoint

def main(config):
    config.device = torch.device('cuda:{}'.format(config.gpu)
                                 if torch.cuda.is_available() else 'cpu')

    # load data_loader
    data_loader = get_dataloader(config)
    check_point = Checkpoint(config)
    operator = Operator(config, check_point)

    if config.is_train:
        operator.train(data_loader)
    else:
        operator.test(data_loader)


if __name__ == "__main__":
    config = get_config()
    main(config)
