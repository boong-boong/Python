import cv2
import pandas as pd
import numpy as np
import os

dir_path = './data'
img_size = 28

os.makedirs('./train/', exist_ok=True)
os.makedirs('./test/', exist_ok=True)

df_dict_train = {
    'file_name': [],
    'label': []
}
df_dict_test = {
    'file_name': [],
    'label': []
}

for idx, dir_name in enumerate(os.listdir(dir_path)):
    # print(idx, dir_name)
    file_path = dir_path + '/' + dir_name + '/'
    for file in os.listdir(file_path)[:100]:
        cv2.imread
        img = cv2.imread(file_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(f'./train/{dir_name+file}', img)
        df_dict_train['file_name'].append(dir_name+file)
        df_dict_train['label'].append(idx)

    for file in os.listdir(file_path)[100:]:
        img = cv2.imread(file_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(f'./test/{dir_name+file}', img)
        df_dict_test['file_name'].append(dir_name+file)
        df_dict_test['label'].append(idx)

df = pd.DataFrame(df_dict_train)
# print(df)
df.to_csv('./annotation_train.csv')
df = pd.DataFrame(df_dict_test)
# print(df)
df.to_csv('./annotation_test.csv')
