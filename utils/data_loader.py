import os
import sys
from itertools import count
from collections import namedtuple, defaultdict

import numpy as np
import torch
import torch.utils.data as data

class KBDataset(data.Dataset):
    """docstring for KBDataSet."""
    def __init__(self, filename, is_cuda):
        super(KBDataset, self).__init__()
        self.filename = filename
        self.triplets = []
        self.is_cuda = is_cuda
        self.read_data()

    def __getitem__(self, index):
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)

    def read_data(self):
        with open(self.filename) as f:
            size = f.readline()
            for ln in f.readlines():
                s, t, r = ln.strip().split()
                src = torch.LongTensor([int(s)])
                rel = torch.LongTensor([int(r)])
                dst = torch.LongTensor([int(t)])
                if self.is_cuda:
                    src = src.cuda()
                    rel = rel.cuda()
                    dst = dst.cuda()
                self.triplets.append([src, rel, dst])

if __name__ == '__main__':
    # load data
    task_dir = "FB15K237"
    # init dataset
    train_data = KBDataset(os.path.join("../data", task_dir, 'train2id.txt'), False)
    valid_data = KBDataset(os.path.join("../data", task_dir, 'valid2id.txt'), False)
    test_data = KBDataset(os.path.join("../data", task_dir, 'test2id.txt'), False)

    # init data loader
    train_loader = data.DataLoader(train_data, batch_size=3, shuffle=True)
    valid_loader = data.DataLoader(valid_data, batch_size=3, shuffle=True)
    test_loader = data.DataLoader(valid_data, batch_size=3, shuffle=True)
