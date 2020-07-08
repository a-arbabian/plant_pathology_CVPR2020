import torch
from tqdm import tqdm
from dataset import CatsDogsDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, CenterCrop, Normalize, ToTensor, Resize
from torchvision.models import mobilenet_v2, resnet18
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

ROOT_DIR = '/home/ali/Desktop/Code/CatsDogsCNN/Cats-Dogs/PetImages'
IMG_SIZE = 224
EPOCHS = 20
## PR: BCELoss: 0.012183905643756496

#int(IMG_SIZE + (0.15 * IMG_SIZE))
train_transforms = transforms.Compose([Resize((IMG_SIZE, IMG_SIZE)),
                                       # CenterCrop(IMG_SIZE),
                                       # ColorJitter(0.1, 0.1, 0.04, 0.03),
                                       ToTensor(),
                                       #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
                                       ])


dataset = CatsDogsDataset(ROOT_DIR, transform=train_transforms)
train_loader = DataLoader(dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=8,
                          drop_last=True)

model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)

criterion = BCEWithLogitsLoss(reduction='mean')
optimizer = Adam(model.parameters(), lr=0.001)

model.train()
model.cuda()

for epoch in range(EPOCHS):
    running_loss = 0.

    for sample in tqdm(train_loader):
        inputs = sample['image'].cuda()
        labels = sample['label'].unsqueeze(1).float().cuda()

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

    print(f"BCELoss: {running_loss/len(train_loader)}")

# def train(model, dataloader, criterion, optimizer):
