import os
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, resnet18, resnext50_32x4d
from torchvision.transforms import Normalize, ToTensor, Resize, Compose

from PIL import ImageFilter
import numpy as np
import pandas as pd

from dataset import PlantDataset


if __name__ == "__main__":
    # TODO: Add arg parser
    SEED = 1984
    IMG_SIZE = np.array([480, 768], dtype=int) // 2  # Dataset images are size (2048, 1365)
    CROP_SIZE = np.array(IMG_SIZE * 1.2, dtype=int)
    EPOCHS = 30

    ROOT_DIR = '/home/ali/Documents/Datasets/plant_pathology_2020_FGVC7/images'
    CSV_PATH = '/home/ali/Documents/Datasets/plant_pathology_2020_FGVC7/test.csv'
    test_csv = pd.read_csv(CSV_PATH, index_col='image_id')
    probas = pd.DataFrame(index=test_csv.index, columns=['healthy',
                                                         'multiple_diseases',
                                                         'rust',
                                                         'scab'])


    val_transforms = Compose([Resize(IMG_SIZE),
                              ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                              ])


    eval_dataset = PlantDataset(ROOT_DIR, csv_path=CSV_PATH, transform=val_transforms)
    print(f"Testing on {len(eval_dataset)} samples!")

    eval_loader = DataLoader(eval_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4,
                             drop_last=True)

    model = resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.cuda()
    model.eval()

    model_weights_path = "/home/ali/Desktop/Code/plant_pathology_CVPR2020/logs/11072020_133639/best.pt"
    csv_out_path = "/home/ali/Desktop/Code/plant_pathology_CVPR2020/logs/11072020_133639/submission.csv"

    weights = torch.load(model_weights_path)
    model.load_state_dict(weights, strict=True)

    running_correct = 0
    # TODO: make evaluation batched
    for sample in tqdm(eval_loader):
        image_id = sample['image_id'][0]
        inputs = sample['image'].cuda()
        if 'label' in sample.keys():
            labels = sample['label'].cuda()

        with torch.no_grad():
            outputs = model(inputs)
            # softmax across logits
            softmax = Softmax(dim=1)
            preds = softmax(outputs.cpu())
            # add preds to submission CSV
            probas.at[image_id, 'healthy'] = preds.squeeze().numpy()[0]
            probas['multiple_diseases'][image_id] = preds.squeeze().numpy()[1]
            probas['rust'][image_id] = preds.squeeze().numpy()[2]
            probas['scab'][image_id] = preds.squeeze().numpy()[3]

    probas.to_csv(csv_out_path)
    print(f"CSV saved to: {csv_out_path}!")
