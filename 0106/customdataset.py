import glob
import os
from torch.utils.data import Dataset
import cv2


class my_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.all_image_path = glob.glob(os.path.join(path, '*', '*.png'))
        self.transform = transform
        self.label_dict = {
            'paper': 0,
            'rock': 1,
            'scissors': 2
        }

    def __getitem__(self, item):
        image_path = self.all_image_path[item]
        label = image_path.split('\\')[1]
        label = self.label_dict[label]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label

    def __len__(self):
        return len(self.all_image_path)


if __name__ == '__main__':
    test = my_dataset('./dataset/train/', transform=None)
    for i in test:
        pass
