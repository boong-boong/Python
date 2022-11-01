import keras
print(keras.__version__)

from keras.datasets import imdb
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000) # 상위 단어 만개

train_data.shape #1차원
train_data[0] # 내부적으론 글자
train_labels[0] # 1이면 긍정 0이면 부정 - 이진 분류 문제 activation 함수 sigimoid 사용

word_index = imdb.get_word_index() # 각 번호가 원래 무슨 단어였는지
#word_index
reverse_word_index = dict([value, key] for (key, value) in word_index.items())

decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]]) # 0,1,2는 기호로 되어 있어 제외, 그렇게 없다면 ?로 치환시킴
decoded_review

# 데이터 준비
import numpy as np

def vectorize_sequences(sequences, dimension = 10000):
  results = np.zeros((len(sequences),dimension))

  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1

  return results
# Data의 Encoding
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train[0])

# float type으로 변환
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 신경망의 구축
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# from tensorflow.keras import optimizers
from keras import optimizers # optimizer의 제약을 걸기위해 사용
model.compile(optimizer = optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy']) # optimizer = adama, rmsprop중 선택 이진분류라면 loss='binary_crossentropy'

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:] # train, validation, test 3가지로 나누고 학습하면서 검증, 마지막에 테스트해서 정확도 높임

history = model.fit(partial_x_train, partial_y_train, epochs=20,batch_size = 256, validation_data=(x_val, y_val)) # 학습시키고 epochs를 바꾸면 학습을 더 시키는 효과가 나옴, 모델을 다시 실행시켜야 함

# 실험 결과 데이터를 가져온다.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

# 실험결과 시각화
import matplotlib.pyplot as plt
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()

plt.plot(epochs, acc,"bo", label = 'Training acc')
plt.plot(epochs, val_acc,"b-", label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

'''
for i, c in enumerate(['a','b','c'],start = 1):
  print(i, c)

temp = enumerate(['A',"B",'C'])
next(temp)
next(temp)
#type(temp)
#list(temp)
'''