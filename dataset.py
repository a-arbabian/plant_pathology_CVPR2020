import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random

class PlantDataset(Dataset):

    def __init__(self, root_dir, csv_path, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_path (string): Path to CSV file with image names and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.csv = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def check(self):
        idx = random.randint(0, len(self.csv))
        im_name = self.csv['image_id'][idx]

        fp = os.path.join(self.root_dir, im_name) + '.jpg'
        # label = self.master_list[idx][1]

        img = Image.open(fp).convert('RGB')
        plt.imshow(img)
        # plt.title(f'{img.size}') #(2048, 1365)
        label = [self.csv["healthy"][idx],
                 self.csv["multiple_diseases"][idx],
                 self.csv["rust"][idx],
                 self.csv["scab"][idx]]
        plt.title(f'{label}')
        plt.show()

    def __getitem__(self, idx):
        im_name = self.csv['image_id'][idx]
        fp = os.path.join(self.root_dir, im_name) + '.jpg'
        img = Image.open(fp).convert('RGB')

        #(2048, 1365)
        label = torch.FloatTensor([self.csv["healthy"][idx],
                                   self.csv["multiple_diseases"][idx],
                                   self.csv["rust"][idx],
                                   self.csv["scab"][idx]])

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'label': label}
        return sample

