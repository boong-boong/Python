import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])

x = np.array(3)
print(x)
print(x.shape)
print(np.ndim(x))

# 벡터 (1차원 텐서)
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
c = a + b

print(c)
print(c.shape)
print(np.ndim(c))

c = a * b
print(c)
print(c.shape)
print(np.ndim(c))

# 스칼라와 벡터의 곱
a = np.array(10) # 스칼라
b = np.array([1,2,3]) # 1차원 텐서
c = a * b

print(c)

# 전치행렬

A = np.array([[1,2,3],[4,5,6]])
print('A\n', A)
print('A.shape\n', A.shape)
print('---------------')

A_ = A.T
print('A_\n', A_)
print('A_.shape\n', A_.shape)
print('---------------')

from keras.datasets import mnist # 실행시 인터넷에서 데이터를 다운 받음
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

print(train_images.ndim)

print(train_images.shape)

print(train_images.dtype)

temp_image = train_images[3]
plt.imshow(temp_image, cmap='gray')

train_labels[3]

# 번외편 PKL 파일 저장 및 로딩

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import joblib

dataset = datasets.load_iris()

X, y = dataset['data'], dataset['target']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

joblib.dump(model, './knn_model.pkl')

loaded_model = joblib.load('./knn_model.pkl')
#loaded_model.predict
score = loaded_model.score(X, y)
print('정확도: {score:.3f}'.format(score=score))