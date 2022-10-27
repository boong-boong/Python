from sklearn.datasets import load_iris
iris_dataset = load_iris()

# print(iris_dataset.DESCR)
# iris_dataset['DESCR']

# 데이터 확인
print('iris_dataset의 키:\n', iris_dataset.keys())

print('타겟의 이름:\n', iris_dataset['target_names'])
print('특징의 이름:\n', iris_dataset['feature_names'])

print('data의 타입:\n', type(iris_dataset['data']))

print('data의 크기:\n', iris_dataset['data'].shape)

print('data의 처음 다섯개:\n', iris_dataset['data'][:5])

print('target의 타입:', type(iris_dataset['target']))

print('target의 크기:', iris_dataset['target'].shape)

print('target:\n', iris_dataset['target'])

#데이터의 준비
from sklearn.model_selection import train_test_split # 데이터를 train, test로 쪼개줌
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0, test_size=0.3)
# 학습 데이터, 라벨, 테스트용 데이터, 라벨로 나눔, test_size 1기준 비율 설정

print('x_train 크기:',x_train.shape)
print('y_train 크기:',y_train.shape)
print('x_test 크기:',x_test.shape)
print('y_test 크기:',y_test.shape)

# dataframe 객체로 만들기, 데이터의 시각화
import pandas as pd
import matplotlib.pyplot as plt


iris_dataframe = pd.DataFrame(x_train, columns = iris_dataset['feature_names'])
#print(iris_dataframe)
pd.plotting.scatter_matrix(iris_dataframe, figsize=(10,10), marker='o', c = y_train, cmap='viridis', alpha = 0.8)
#plt.show()

# 첫 번째 머신 러닝 모델: k-최근접 알고리즘(KNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)

# 예측하기
import numpy as np
x_new = np.array([[5, 2.9, 1, 0.2]]) # 임의의 꽃받침 길이 넓이, 예측 데이터, 우선 [[]]로 2차원 데이터로 사용
prediction = knn.predict(x_new)

print('예측:', prediction)
print('예측한 타겟의 이름:', iris_dataset['target_names'][prediction])

#모델의 평가
y_pred = knn.predict(x_test)
print('테스트 세트에 대한 예측값:\n', y_pred)

print(np.mean(y_pred == y_test)) # mean = 평균
print('테스트 세트에 대한 정확도:\n', np.mean(y_pred == y_test)*100)
print('테스트 세트에 대한 정확도:\n', knn.score(x_test, y_test))