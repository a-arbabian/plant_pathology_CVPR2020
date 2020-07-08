import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


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
        self.csv = root_dir
        self.transform = transform

        cat_filenames = os.listdir(os.path.join(root_dir, 'Cat'))
        dog_filenames = os.listdir(os.path.join(root_dir, 'Dog'))
        self.cat_list = [(os.path.join(root_dir, "Cat", fn), 0) for fn in cat_filenames if '.jpg' in fn]
        self.dog_list = [(os.path.join(root_dir, "Dog", fn), 1) for fn in dog_filenames if '.jpg' in fn]
        self.master_list = self.cat_list + self.dog_list

    def __len__(self):
        return len(self.master_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fp = self.master_list[idx][0]
        label = self.master_list[idx][1]

        img = Image.open(fp).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # plt.imshow(img)
        # plt.show()

        # if img.shape[0] != 3:
        #     img = img.repeat(3, 1, 1)

        sample = {'image': img, 'label': label}
        return sample

