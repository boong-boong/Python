# train loop
# val loop
# 모델 save
# 평가 함수
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torchvision import models



# 평가 함수
def calculate_acc(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((output == target).sum(dim=0), output.size(0)).item()


def save_model(model, save_dir, file_name='last.pt'):
    # save model
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print('멀티 GPU 저장')
        torch.save(model.module.state_dict(), output_path)
    else:
        print('싱글 GPU 저장')
        torch.save(model.state_dict(), output_path)

def train(train_loader, model, criterion, optimizer, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (image, target) in enumerate(stream):
        images = image.to(device)
        targets = target.to(device).view(-1, 1)

        output = model(images)
        loss = criterion(output, targets.float())
        accuracy = calculate_acc(output, targets)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch : {epoch}. Train.     {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )

def validate(val_loader, model, criterion, epoch, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(stream):
            images = image.to(device)
            targets = target.to(device).view(-1, 1)

            output = model(images)
            loss = criterion(output, targets.float())
            accuracy = calculate_acc(output, targets)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)

            stream.set_description(
                "Epoch : {epoch}. val.     {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor)
            )

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name} : {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
