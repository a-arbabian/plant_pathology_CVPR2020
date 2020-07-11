import os
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, resnet18, resnext50_32x4d
from torchvision.transforms import ColorJitter, CenterCrop, Normalize, ToTensor, Resize, Compose, RandomHorizontalFlip,\
    RandomAffine, RandomRotation, RandomChoice, RandomApply, Lambda
from torch.utils.tensorboard import SummaryWriter

from PIL import ImageFilter
import numpy as np
import pandas as pd
from apex import amp

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from dataset import PlantDataset
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


def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    running_loss = 0.
    for epoch in range(num_epochs):
        for sample in dataloader:
            inputs = sample['image'].cuda()
            labels = sample['label'].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()

    train_loss = running_loss / (len(dataloader) * num_epochs)
    return train_loss


def validate(model, dataloader, criterion, total_set_size):
    model.eval()
    running_loss = 0.
    running_correct = 0

    for sample in dataloader:
        inputs = sample['image'].cuda()
        labels = sample['label'].cuda()

        with torch.no_grad():
            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Update loss and accuracy
            running_loss += loss.item()
            # softmax across logits
            softmax = Softmax(dim=1)
            preds = softmax(outputs)
            # argmax so both tensors are no longer one-hot
            preds_argmax = preds.argmax(dim=1)
            labels_argmax = labels.argmax(dim=1)
            # add correct preds to running total
            running_correct += (preds_argmax == labels_argmax).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = running_correct / total_set_size

    return val_loss, val_acc


def train_one_fold(i_fold, model, criterion, optimizer, train_loader, val_loader):
    # Train
    train_fold_results = []
    for epoch in range(EPOCHS):

        print('  Epoch {}/{}'.format(epoch + 1, EPOCHS))
        print('  ' + ('-' * 20))
        # os.system(f'echo \"  Epoch {epoch}\"')

        model.train()
        tr_loss = 0

        for step, sample in enumerate(tqdm(train_loader)):
            images = sample['image']
            labels = sample['label']

            images = images.cuda().float()
            labels = labels.cuda().float()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, sample in enumerate(tqdm(val_loader)):

            images = sample['image']
            labels = sample['label']

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            images = images.cuda().float()
            labels = labels.cuda().float()

            with torch.no_grad():
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.softmax(outputs, dim=1).data.cpu()

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)

        train_fold_results.append({
            'fold': i_fold,
            'epoch': epoch,
            'train_loss': tr_loss / len(train_loader),
            'valid_loss': val_loss / len(val_loader),
            'valid_score': roc_auc_score(val_labels, val_preds, average='macro'),
        })

    return val_preds, train_fold_results


def blur_gauss(img):
    return img.filter(ImageFilter.GaussianBlur(3))

if __name__ == "__main__":
    # TODO: Add arg parser
    N_FOLDS = 5
    SEED = 1984
    IMG_SIZE = np.array([480, 768], dtype=int) // 2  # Dataset images are size (2048, 1365)
    CROP_SIZE = np.array(IMG_SIZE * 1.2, dtype=int)
    EPOCHS = 30

    ROOT_DIR = '/home/ali/Documents/Datasets/plant_pathology_2020_FGVC7/images'
    CSV_PATH = '/home/ali/Documents/Datasets/plant_pathology_2020_FGVC7/train.csv'
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    print(f"Training on {len(train_df)} samples!")
    print(f"Validating on {len(val_df)} samples!")

    imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
    imagenet_mean = tuple(imagenet_mean.astype(int))

    train_transforms = Compose([RandomApply([Lambda(blur_gauss)], p=0.2),
                                Resize(CROP_SIZE),
                                CenterCrop(IMG_SIZE),
                                RandomHorizontalFlip(),
                                RandomAffine(degrees=20, translate=(0.1, 0.1), shear=20, resample=2, fillcolor=imagenet_mean),
                                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                                ])

    val_transforms = Compose([Resize(IMG_SIZE),
                              ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                              ])

    train_dataset = PlantDataset(ROOT_DIR, csv=train_df, transform=train_transforms)
    val_dataset = PlantDataset(ROOT_DIR, csv=val_df, transform=val_transforms)

    train_loader = DataLoader(train_dataset,
                              batch_size=24,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=24,
                            shuffle=False,
                            num_workers=4,
                            drop_last=True)

    model = resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.cuda()

    criterion = CrossEntropyLossOneHot()
    optimizer = Adam(model.parameters(), lr=0.001)
    # TODO: Warmup lr scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     factor=0.1,
                                     patience=2,
                                     mode='min',
                                     verbose=True)

    # Mixed precision with amp
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Tensorboard setup
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    writer = SummaryWriter(f"./logs/{dt_string}/")

    # TODO: block needs to be fixed for KFolds set-up
    # folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
    #     print("Fold {}/{}".format(i_fold + 1, N_FOLDS))
    #
    #     valid = train_df.iloc[valid_idx]
    #     valid.reset_index(drop=True, inplace=True)
    #
    #     train = train_df.iloc[train_idx]
    #     train.reset_index(drop=True, inplace=True)
    #
    #     dataset_train = PlantDataset(df=train, transforms=transforms_train)
    #     dataset_valid = PlantDataset(df=valid, transforms=transforms_valid)
    #
    #     dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    #     dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    #
    #     device = torch.device("cuda:0")
    #
    #     model = PlantModel(num_classes=4)
    #     model.to(device)
    #
    # for epoch in range(EPOCHS):
    #     for i_fold in range(N_FOLDS):
    #         val_preds, train_fold_results = train_one_fold(i_fold,
    #                                                        model,
    #                                                        criterion,
    #                                                        optimizer,
    #                                                        train_loader,
    #                                                        val_loader)
    best_val_loss = np.inf
    for epoch in tqdm(range(EPOCHS), desc="Epoch: "):
        train_loss = train(model, train_loader, criterion, optimizer, num_epochs=1)
        writer.add_scalar('loss/train', train_loss, epoch)

        val_loss, val_acc = validate(model, val_loader, criterion, total_set_size=len(val_dataset))
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('acc/val', val_acc, epoch)

        writer.add_scalar('lr/train', optimizer.param_groups[0]['lr'], epoch)
        lr_scheduler.step(val_loss)
        writer.flush()

        if val_loss < best_val_loss:
            print(f"Record val loss: {val_loss}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./logs/{dt_string}/best.pt")

    writer.close()


