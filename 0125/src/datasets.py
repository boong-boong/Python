import torch
import cv2
import numpy as np
import os
import glob

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform


class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transform=None):
        self.dir_path = dir_path
        self.width = width
        self.height = height
        self.classes = classes
        self.transform = transform

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f'{self.dir_path}/*.jpg')
        self.all_images = [image_path.split('/')[1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getattr__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        image = cv2.imread(image_path)
        image = image.astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            xmin = int(member.find('boundbox').find('xmin').text)
            xmax = int(member.find('boundbox').find('xmax').text)
            ymin = int(member.find('boundbox').find('ymin').text)
            ymax = int(member.find('boundbox').find('ymax').text)

            xmin_final = (xmin / image_width)*self.width
            xmax_final = (xmax / image_width)*self.width
            ymin_final = (ymin / image_height)*self.height
            ymax_final = (ymax / image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])



    def __len__(self):
        pass
