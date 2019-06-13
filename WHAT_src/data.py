import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_dataloader(config):
    data_dir = config.data_dir
    batch_size = config.batch_size

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

    if config.data_name == 'mnist':
        train_dataset = dset.MNIST(root=data_dir, train=True, transform=trans, download=True)
        test_dataset = dset.MNIST(root=data_dir, train=False, transform=trans, download=True)
    elif config.data_name == 'fashion_mnist':
        train_dataset = dset.FashionMNIST(root=data_dir, train=True, transform=trans, download=True)
        test_dataset = dset.FashionMNIST(root=data_dir, train=False, transform=trans, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    data_loader = {'train': train_loader, 'test': test_loader}

    return data_loader
