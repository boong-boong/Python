import os.path
import sys

import torch
import torchvision.models
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from utils import aug_function
from torch.utils.data import DataLoader
from customdataset import customDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_aug = aug_function(mode_flag='train')
val_aug = aug_function(mode_flag='test')

train_dataset = customDataset('./data04/train/', transform=train_aug)
val_dataset = customDataset('./data04/test/', transform=val_aug)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

finetune_net = torchvision.models.efficientnet_b3(pretrained=True)
# print(finetune_net)
finetune_net.classifier[1] = nn.Linear(1536, 2)
finetune_net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(finetune_net.parameters(), lr=0.0003,
                              weight_decay=0.0005)

epochs = 50
best_val_acc = 0.0

train_steps = len(train_loader)
valid_step = len(val_loader)
save_path = 'best.pt'

dfForAccuracy = pd.DataFrame(index=list(range(epochs)), columns=['Epoch', 'Accuracy'])

if os.path.exists(save_path):
    best_val_acc = max(pd.read_csv('./modelAccuracy.csv')['Accuracy'].tolist())
    finetune_net.load_state_dict(torch.load(save_path))

for epoch in range(epochs):
    running_loss = 0
    val_acc = 0
    train_acc = 0

    finetune_net.train()
    train_bar = tqdm(train_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(train_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = finetune_net(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = f'train epoch[{epoch + 1} / {epoch}], loss{loss.data:.3f}'

    finetune_net.eval()
    with torch.no_grad():
        valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
        for data in valid_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = finetune_net(images)
            val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

    val_accuracy = val_acc / len(val_dataset)
    train_accuracy = train_acc / len(train_dataset)

    dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
    dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 3)
    print(f'epoch [{epoch + 1}/{epochs}]'
          f'train loss: {(running_loss / train_steps):.3f}'
          f'train_acc: {train_accuracy:.3f} val_acc: {val_accuracy:.3f}'
          )

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(finetune_net.state_dict(), save_path)

    if epoch == epochs - 1:
        dfForAccuracy.to_csv('./modelAccuracy.csv', index=False)
