import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import warnings
warnings.filterwarnings('ignore') # 경고 무시

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
label = iris.target
columns = iris.feature_names

data = pd.DataFrame(data, columns=columns)
print(data.head())
print(data.shape)

# 데이터 준비
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.2, random_state= 2022)

# Logistic Regression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print('로지스틱 회귀, 정확도: {:.2f}'.format(accuracy_score(y_test,y_pred)))

# Support Vector Machine
from sklearn.svm import SVC
svc = SVC(C=100)

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

print('서포트 벡터 머신, 정확도: {:.2f}'.format(accuracy_score(y_test,y_pred)))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print('결정 트리, 정확도: {:.2f}'.format(accuracy_score(y_test,y_pred)))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('랜덤 포레스트, 정확도: {:.2f}'.format(accuracy_score(y_test,y_pred)))
# 어떤 알고리즘을 써도 결과가 비슷함 -> 이 의미는 데이터가 좋은것, 3퍼 정도는 데이터 자체의 오류로 볼수있음