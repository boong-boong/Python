from keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images.shape
len(train_labels)
test_images.shape

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) # 2차원 형태로 넣어줘야되기 때문에 ((28*28),)
network.add(layers.Dense(10, activation='softmax')) # 다중 분류 문제라 softmax사용

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 데이터 타입의 변환
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255 # 학습에선 정수형은 좋지 못함
#print(train_images[0])
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

# 분류형 데이터의 설정
# from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels) # 평가 loss, accuracy
print('test_acc: ', test_acc)