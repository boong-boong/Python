import keras
import numpy as np
import matplotlib.pyplot as plt

# 로이터 기사 데이터 셋을 로딩

from keras.datasets import reuters
(train_data, train_labels),(test_data, test_labels) = reuters.load_data(num_words = 10000)

print(train_data.shape)
print(train_data[0])

word_index = reuters.get_word_index()
reverse_word_index = dict([value, key] for (key, value) in word_index.items())

decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

print(decoded_newswire)
print(train_labels[0])

# 데이터의 준비
def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences),dimension))

  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  
  return results

#데이터의 변환
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

train_labels

# 라벨 데이터의 Encoding
def to_one_hot(labels, dimension = 46):
  results = np.zeros((len(labels),dimension))

  for i, sequence in enumerate(labels):
    results[i, sequence] = 1
  
  return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# categoraical 데이터로 변환
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels) # 원핫, 카테고리컬 둘중 선택, 둘다 하는경우도 있음
one_hot_test_labels = to_categorical(test_labels)

# 신경망 구성
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(1000, activation='relu')) # hidden layer를 4, 1000 바꿔봄, 작으면 정보의 소실, 크면 불필요하고 느려짐
model.add(layers.Dense(46, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
# 훈련 데이터의 준비

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size=512, validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(loss)+1)

# 결과를 시각화
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()