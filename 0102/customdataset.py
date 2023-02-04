import os
import glob
from PIL import Image

from torch.utils.data import Dataset


class customDataset(Dataset):
    def __init__(self, path, transform=None):
        self.all_file_path = glob.glob(os.path.join(path, '*', '*.png'))
        self.transform = transform
        self.label_dict = {'cloudy': 0, 'desert': 1, 'green_area': 2, 'water': 3}
        # self.img_list = []
        # for img_path in self.all_file_path:
        #     self.img_list.append(Image.open(img_path))
        #     # 초기화하면서 오픈했기 때문에 메모리 많이 필요 대신 getitem 빨라짐

    def __getitem__(self, item):
        # img = self.img_list[item]
        # print(img)
        img_path = self.all_file_path[item]
        label_temp = img_path.split('\\')
        label = self.label_dict[label_temp[1]]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.all_file_path)


test = customDataset('./dataset/train', transform=None)
#
# for i in test:
#     print(i)
