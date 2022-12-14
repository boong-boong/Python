from torch.utils.data.dataset import Dataset
import pandas as pd


class MyCustomDataset(Dataset):
    def __init__(self, path):
        self.readFile = pd.read_csv(path)

    def __getitem__(self, index):
        filename = self.readFile.iloc[index, 1]
        bbox = self.readFile.iloc[index, 2]

        return filename, bbox

    def __len__(self):
        return len(self.readFile)


temp = MyCustomDataset("./box.csv")

for i in temp:
    print(i)