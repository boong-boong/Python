import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

img_dir = '/Users/leejuhyung/Desktop/AI/Python/Image Processing/1209/data/FashionMNIST/test'
mnist_data = pd.read_csv('/Users/leejuhyung/Desktop/AI/Python/Image Processing/1209/data/FashionMNIST/annotation_test.csv',
                         names=['file_name', 'labels'], skiprows=[0])

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}

print(len(mnist_data))
print(len(os.listdir(img_dir)))

# for i in range(len(mnist_data)):
for i in range(5):  # 너무 많으니 5개만 출력
    file_name, label = mnist_data.iloc[i]
    img = cv2.imread(os.path.join(img_dir, file_name))
    plt.subplot(1, 5, i+1)
    plt.imshow(img, 'gray')
    plt.title(labels_map[label])
plt.show()
