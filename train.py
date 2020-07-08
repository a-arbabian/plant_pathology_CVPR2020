import torch
import numpy as np
from tqdm import tqdm
from dataset import PlantDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, CenterCrop, Normalize, ToTensor, Resize
from torchvision.models import mobilenet_v2, resnet18, resnext50_32x4d
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from loss import CrossEntropyLossOneHot

# from albumentations import (
#     Compose,
#     GaussianBlur,
#     HorizontalFlip,
#     MedianBlur,
#     MotionBlur,
#     Normalize,
#     OneOf,
#     RandomBrightness,
#     RandomContrast,
#     Resize,
#     ShiftScaleRotate,
#     VerticalFlip,
# )

ROOT_DIR = '/home/ali/Documents/Datasets/plant_pathology_2020_FGVC7/images'
CSV = '/home/ali/Documents/Datasets/plant_pathology_2020_FGVC7/train.csv'
IMG_SIZE = np.array([480, 768], dtype=int) // 2
EPOCHS = 20

train_transforms = transforms.Compose([Resize(IMG_SIZE),
                                       ToTensor()])

dataset = PlantDataset(ROOT_DIR, csv_path=CSV, transform=train_transforms)
# for i in range(100):
#     dataset.check()







train_loader = DataLoader(dataset,
                          batch_size=16,
                          shuffle=True,
                          num_workers=8,
                          drop_last=True)

model = resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)

criterion = CrossEntropyLossOneHot()
optimizer = Adam(model.parameters(), lr=0.001)

model.train()
model.cuda()

for epoch in range(EPOCHS):
    running_loss = 0.

    for sample in tqdm(train_loader):
        inputs = sample['image'].cuda()
        labels = sample['label'].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        #print(loss)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print(f"CELoss: {running_loss/len(train_loader)}")

# def train(model, dataloader, criterion, optimizer):
