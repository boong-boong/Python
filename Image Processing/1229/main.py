import os
import glob
from custom_dataset import custom_dataset
from torch.utils.data import DataLoader
from torchvision import models

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_data = custom_dataset('D:\\data\\train')
test_data = custom_dataset('D:\\data\\test')

# dataloader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# model call
net = models.__dict__['resnet18'](pretrained=True)