from tifffile import imread

import numpy as np
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
                    self.label.append(round(float(dis), 2))
                    # if len(self.spa) == 8:
                    #     break
        elif typeset == 'val':
            with open('./dataset/val.txt', 'r') as f:
                for line in f:
                    path, dis = line.strip().split()
                    self.spa.append(path)
                    self.label.append(round(float(dis), 2))
        self.ang = [x.replace('spatial', 'angular') for x in self.spa]

    def __len__(self):
        return len(self.spa)

    def __getitem__(self, index):
        imspa = imread(self.spa[index]).astype(np.float32) / 65535
        imang = imread(self.ang[index]).astype(np.float32) / 65535
        return np.expand_dims(imspa, axis=1), np.expand_dims(imang, axis=1), self.label[index]