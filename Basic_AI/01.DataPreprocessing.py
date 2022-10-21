import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn

import os
from os.path import join


abalone_path = join('.', 'Basic_AI/abalone.txt')
column_path = join('.', 'Basic_AI/abalone_attributes.txt')

abalone_columns = list()

for line in open(column_path):
    abalone_columns.append(line.strip())

print(abalone_columns)
data = pd.read_csv(abalone_path, header=None, names=abalone_columns)

label = data['Sex']
del data['Sex']  # 성별 항목 삭제

#print(label)
#print(data)

print(data.describe())  # 전체 컬럼의 요약 정보, 빠진 데이터 확인용으로 확인 가능
print(data.info())

# scaling
# Min-Max Scaling
'''data = (data - np.min(data))/(np.max(data) - np.min(data))
print(data)'''

from sklearn.preprocessing import MinMaxScaler # 결과가 numpy배열로 나오는게 차이점, 보통 이걸로 쓸것
mMscaler = MinMaxScaler()
#fit() 데이터를 맞춰보는 과정, 데이터 변환을 위한 기준 정보를 설정하는 역할
mMscaler.fit(data)
#transform() 설정된 정보를 이용하여 데이터를 변환 하는 과정
mScaled_data = mMscaler.transform(data)
print(mScaled_data)
# mScaled_data = mMscaler.fit_transform(data) 한줄로 처리

mScaled_data = pd.DataFrame(mScaled_data, columns=data.columns) # dataframe으로 변환
print(mScaled_data)

# Standard Scaling
from sklearn.preprocessing import StandardScaler
sdscaler = StandardScaler()

sdscaled_data = sdscaler.fit_transform(data)
sdscaled_data = pd.DataFrame(sdscaled_data, columns=data.columns)
print(sdscaled_data)

# Sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ros = RandomOverSampler()
rus = RandomUnderSampler()

oversampled_data, oversampled_label = ros.fit_resample(data, label)
undersampled_data, undersampled_label = rus.fit_resample(data, label)

oversampled_data = pd.DataFrame(oversampled_data, columns = data.columns)
undersampled_data = pd.DataFrame(undersampled_data, columns = data.columns)

print('원본 데이터의 클래스 비율: \n{}'.format(pd.get_dummies(label).sum()))
print('oversampled 데이터의 클래스 비율: \n{}'.format(pd.get_dummies(oversampled_label).sum()))
print('undersampled_ 데이터의 클래스 비율: \n{}'.format(pd.get_dummies(undersampled_label).sum()))
# over, under sampling은 데이터의 편향이 생길수 있음

# SMOTE
from sklearn.datasets import make_classification # 샘플데이터 생성, 전복 데이터셋은 SMOTE샘플을 보기 어려움
data, label = make_classification(n_samples=1000,
                    n_features=2,
                    n_informative=2,
                    n_redundant=0,
                    n_repeated=0,
                    n_classes=3,
                    n_clusters_per_class=1,
                    weights=[0.05,0.15,0.8],
                    class_sep=0.8,
                    random_state=2019)


fig = plt.Figure(figsize=(12,6))
plt.scatter(data[:,0],data[:,1],c=label,alpha=0.3)
#plt.show()

from imblearn.over_sampling import SMOTE
smote = SMOTE()

smoted_data, smoted_label = smote.fit_resample(data, label)

print('원본 데이터의 클래스 비율: \n{}'.format(pd.get_dummies(label).sum()))
print('smote 결과: \n{}'.format(pd.get_dummies(smoted_label).sum()))

fig = plt.Figure(figsize=(12,6))
plt.scatter(smoted_data[:,0],smoted_data[:,1],c=smoted_label,alpha=0.3)
#plt.show()

# 차원 축소(고차원에서는 사람이 데이터를 읽기 힘들어 차원을 축소함, 또한 데이터가 빈 공간이 생기기 떄문에 데이터 해석시 문제 일으킴)

from sklearn.datasets import load_digits # 손글씨 데이터
digits = load_digits()

print(digits.DESCR) # 설명을 볼 수 있음(sklearn)

data = digits.data
label = digits.target

print(data.shape) # (1797,64) 1797개의 64짜리 배열, 컴퓨터 입장에선 64개 차원
print(label)
print(data[0]) #64개 배열
data[0].reshape(8,8) #8,8배열로 변환
#label[0]  0의미

#plt.imshow(data[0].reshape((8,8))) # 8,8 배열 2차원 이미지 보기
print("Label: {}".format(label[0]))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

new_data = pca.fit_transform(data)

print('원본 데이터의 차원 \n{}'.format(data.shape))
print('PCA를 거친 데이터의 차원 \n{}'.format(new_data.shape))

print(new_data[0]) # 2차원
print(data[0]) # 64차원

plt.scatter(new_data[:,0],new_data[:,1], c=label, alpha=0.4)
#plt.show()

# Encoding
data = pd.read_csv(abalone_path, header=None, names=abalone_columns)

label = data['Sex']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

print(label)

encoded_label = le.fit_transform(label) # label을 encoding

print(encoded_label)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #

one_hot_encoded = ohe.fit_transform(label.values.reshape((-1,1)))

print(one_hot_encoded)