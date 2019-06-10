'''
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

FILE_PATH = '/Users/chokiheum/Data/nyu_v2/nyu_depth_v2_labeled.mat'


class NYU_v2(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        self.file_path = file_path

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            depth = torch.from_numpy(f['depths'][idx].astype('float32'))
            image = torch.from_numpy(f['images'][idx].astype('float32'))
            label = torch.from_numpy(f['labels'][idx].astype('float32'))

            sample = {'depth': depth, 'image': image, 'label': label}

            if self.transform:
                sample = self.transform(sample)

        return sample

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            length = len(f['images'])
        return length


if __name__ == '__main__':
    nyu_dataset = NYU_v2(file_path=FILE_PATH)
    dataloader = DataLoader(nyu_dataset,
                            batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched['image'])
