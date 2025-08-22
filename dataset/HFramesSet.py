import os
from os.path import join
from tifffile import imread

from torch.utils.data import Dataset

class Hframes_Interval(Dataset):
    def __init__(self, typeset='train'):
        super().__init__()
        self.spa = list()
        self.ang = list()
        self.label = list()
        if typeset == 'train':
            with open('./dataset/train.txt', 'r') as f:
                for line in f:
                    path, dis = line.strip().split()
                    self.spa.append(path)
                    self.label.append(float(dis))
        elif typeset == 'val':
            with open('./dataset/val.txt', 'r') as f:
                for line in f:
                    path, dis = line.strip().split()
                    self.spa.append(path)
                    self.label.append(float(dis))
        self.ang = [x.replace('spatial', 'angular') for x in self.spa]

    def __len__(self):
        return len(self.spa)

    def __getitem__(self, index):
        return imread(self.spa[index]), imread(self.ang[index]), self.label[index]