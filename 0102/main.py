from customdataset import customDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import torch
import torch.nn as nn
from torch import optim
from utils import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))
])

# train val test dataset
train_dataset = customDataset('./dataset/train', transform=train_transform)
test_dataset = customDataset('./dataset/test', transform=val_transform)
val_dataset = customDataset('./dataset/val', transform=val_transform)

# train val test loader
train_loader = DataLoader(train_dataset, batch_size=126, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=126, shuffle=False)

net = models.resnet18(pretrained=True)
in_feature_val = net.fc.in_features
net.fc = nn.Linear(in_feature_val, 4)
net.to(device)
# print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

train(100, train_loader, val_loader, net, optimizer, criterion, device, save_path='./best.pt')
