import keras
import numpy as np
import matplotlib.pyplot as plt

# 보스턴 주택 가격 데이터셋
from keras.datasets import boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

train_data.shape
test_data.shape

# standard scaling
mean = train_data.mean(axis = 0)

train_data -= mean

# 표준 편차 구하기
std = train_data.std(axis=0) #std가 0이라면 값이 모든 같은것, 돌려봐야 의미 없음

train_data /= std

train_data

mean = test_data.mean(axis=0)
test_data -= mean
std = test_data.std(axis=0)
test_data /= std

# 신경망 생성
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],))) # input은 모양에 맞춰 줘야함 아니면 빈값이 들어가게됨
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) # 예측한 집값 하나만 나오면 됨

    model.compile(optimizer='rmsprop', 
                loss='mse',
                metrics=['mae'])

    return model

# K-folder 검즘 ,데이터의 수가 적을때 사용
k = 4
num_epochs = 20

num_val_samples = len(train_data) // k # 폴더의 사이즈
all_scores = []

for i in range(k):
    print('처리중인 폴드 #',i)

    # 검증 데이터를 준비: k번째 분할
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_labels = train_labels[i * num_val_samples : (i+1) * num_val_samples]

    # 훈련 데이터의 준비
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples], # 처음에는 아무거도 안겨져옴
        train_data[(i+1)*num_val_samples:]], axis=0)# k폴더 구현, 합칠때 축을 지정해야함

    partial_train_labels = np.concatenate(
        [train_labels[:i * num_val_samples],
        train_labels[(i+1)*num_val_samples:]], axis=0)
  
    model = build_model()
    history = model.fit(partial_train_data,
                        partial_train_labels,
                        epochs=num_epochs,
                        batch_size =1,
                        validation_data=(val_data, val_labels),
                        verbose=0) # 총 epochs * 4 번 돌아감

    mae_history = history.history['mae']
    all_scores.append(mae_history)

    #val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0) #verbose 학습중 출력되는 문구 설정
    #all_scores.append(val_mae)

average_mae_history = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]

plt.plot(range(1,len(average_mae_history)+1), average_mae_history) # 전체 반복횟수가 x축
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()