from dataset_temp import custom_dataset
from utils import train
from utils import validate
from utils import save_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import tqdm




# device
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# train aug
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

# val aug
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

# dataset
train_dataset = custom_dataset('./data/train', transform=train_transform)
val_dataset = custom_dataset('./data/val', transform=val_transform)

# for i, t in train_dataset:
#     print(i, t)


# dataloader
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

# for i, (image, target) in enumerate(train_loader):
#     print(i, image, target)

params = {
    "model": "resnet18",
    "lr": 0.001,
    "batch_size": 64,
    "num_workers": 2,
    "epoch": 10
}
# model loader
model = models.__dict__[params["model"]](pretrained=False)
model.fc = nn.Linear(512, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

# train
for epoch in range(1, params['epoch']+1):
    train(train_loader, model, criterion, optimizer, epoch, device)
    validate(val_loader, model, criterion, epoch, device)

save_dir = './save_model'
save_model(model, save_dir)
