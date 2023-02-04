from torch.utils.data import Dataset
import os
import glob
import cv2


class my_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.all_image_path = glob.glob(os.path.join(path, '*', '*.jpg'))
        self.transform = transform
        self.label_dict = {}

        for idx, category in enumerate(sorted(os.listdir(path))):
            self.label_dict[category] = idx


    def __getitem__(self, item):
        image_path = self.all_image_path[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_path.split('\\')[1]
        label = self.label_dict[label]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        # print(image, label)
        return image, label

    def __len__(self):
        return len(self.all_image_path)


if __name__ == '__main__':
    test = my_dataset('./dataset/train/', transform=None)
    for i in test:
        pass
