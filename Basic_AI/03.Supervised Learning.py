from os import P_ALL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import warnings
warnings.filterwarnings('ignore') # 경고 무시

from sklearn.datasets import load_boston
boston = load_boston()

# print(boston.DESCR)

data = boston.data
label = boston.target
columns = boston.feature_names

data = pd.DataFrame(data, columns=columns)
data.shape

# Simple Linear Regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=2022) 

#x_train('RM') # 입력안됨 sklearn은 2차원 이상의 배열을 인식함

# -1, 1 이해 필요 행에 -1들어오면 알아서 맞춰줌, 데이터가 몇개인지 모를떄 이용
X_train['RM'].values.reshape(-1,1) # 4:15 설명 영상

from sklearn.linear_model import LinearRegression
sim_lr = LinearRegression()

sim_lr.fit(X_train['RM'].values.reshape((-1,1)), y_train)

y_pred = sim_lr.predict(X_test['RM'].values.reshape((-1,1)))

# 결과 살펴보기
from sklearn.metrics import r2_score
print('단순 선형 회귀, R2: {:.4f}'.format(r2_score(y_test, y_pred)))

# 결과 시각화 하기 

line_x = np.linspace(np.min(X_test['RM']), np.max(X_test['RM']), 10)
line_y = sim_lr.predict(line_x.reshape(-1,1))

plt.scatter(X_test['RM'], y_test, s=10, c='black')
plt.plot(line_x,line_y, c='red')
plt.legend(['Regression line','Test data sample'], loc='upper left')
#plt.show()

# Multiple Linear Regression
mul_lr = LinearRegression() # 똑같이 LR사용 칼럼을 더 많이 줌, 기존 RM만 줬지만 다 줌
mul_lr.fit(X_train, y_train)

y_pred = mul_lr.predict(X_test)

print('다중 선형 회귀, R2: {:.4f}'.format(r2_score(y_test, y_pred)))


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dt_regr = DecisionTreeRegressor(max_depth = 2) # 몇 단계까지 내려갈것인가 결정

dt_regr.fit(X_train['RM'].values.reshape((-1,1)),y_train)

y_pred = dt_regr.predict(X_test['RM'].values.reshape((-1,1)))

print('단순 결정 트리 회귀 R2: {:.4f}'.format(r2_score(y_test, y_pred))) # depth높다는것은 학습데이터와 밀착, 테스트용 데이터와는 안 맞게됨, 오버피팅

arr= np.arange(1, 11)

best_depth = 0
best_r2 = 0

for depth in arr:
    dt_regr = DecisionTreeRegressor(max_depth = depth)
    dt_regr.fit(X_train['RM'].values.reshape((-1,1)),y_train)
    y_pred = dt_regr.predict(X_test['RM'].values.reshape((-1,1)))
    temp_r2 = r2_score(y_test, y_pred)
    print('단순 결정 트리 회귀 depth={} R2: {:.4f}'.format(depth, temp_r2))

    if best_r2 < temp_r2:
        best_depth = depth
        best_r2 = temp_r2
print('최적의 결과는 depth={} r2={:.4f}'.format(best_depth, best_r2))

# 다중 결정 트리
dt_regr = DecisionTreeRegressor(max_depth=5)
dt_regr.fit(X_train, y_train)
y_pred = dt_regr.predict(X_test)
print('다중 결정 트리 R2: {:.4f}'.format(r2_score(y_test, y_pred)))

# Support Vector machine Regressor
from sklearn.svm import SVR
svm_regr = SVR(C=1)

svm_regr.fit(X_train['RM'].values.reshape(-1,1), y_train)
y_pred = svm_regr.predict(X_test['RM'].values.reshape(-1,1))

print('단순 서포트 머신 회귀 R2: {:.4f}'.format(r2_score(y_test, y_pred)))

#결과의 시각화
line_x = np.linspace(np.min(X_test['RM']), np.max(X_test['RM']),100)
line_y = svm_regr.predict(line_x.reshape(-1,1))

plt.scatter(X_test['RM'],y_test, c='black')
plt.plot(line_x, line_y, c='red')
plt.legend(['Regression line', 'Test data Sample'], loc='upper left')
#plt.show()


svm_regr = SVR(C=20)
svm_regr.fit(X_train, y_train)
y_pred = svm_regr.predict(X_test)

print('다중 서포트 머신 회귀 R2: {:.4f}'.format(r2_score(y_test, y_pred)))

best_svm_num = 0
best_svm_r2 = 0

for i in range(1, 11):
    svm_regr = SVR(C = i)
    svm_regr.fit(X_train,y_train)
    y_pred = svm_regr.predict(X_test)
    
    temp = r2_score(y_test,y_pred)
    
    print('다중 서포트 머신 회귀 C = {} R2 = {:.4f}'.format(i, temp))
    if best_svm_r2 < temp:
        best_svm_r2 = temp
        best_svm_num = i
print('다중 서포트 머신 회귀 최적값 C = {} R2 = {:.4f}'.format(best_svm_num, best_svm_r2))

# Multi Layer Perceptron Regressor
from sklearn.neural_network import MLPRegressor
mlp_regr = MLPRegressor(solver= 'adam', hidden_layer_sizes=300, max_iter=400) #lbfgs, sgd, adam

mlp_regr.fit(X_train, y_train)
y_pred = mlp_regr.predict(X_test)

print('다중 MLP 회귀, R2: {:.4f}'.format(r2_score(y_test,y_pred)))