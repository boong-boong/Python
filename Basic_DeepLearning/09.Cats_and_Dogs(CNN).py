from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os
import shutil

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import models

# Original Data Path
original_dataset_dir = './datasets/train'

# Small Dataset Path
base_dir = './datasets/cats_and_dogs_small'

if os.path.exists(base_dir):  # 폴더명이 존재하면 지우고 새로 만듬
    shutil.rmtree(base_dir)

os.mkdir(base_dir)

# Train, Validation, Test data
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

# file copy

fnames = []
# for i in range(1000):
#     filename = 'cat.{}.jpg'.format(i)
#     fnames.append(filename)

# cat train data copy
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# dog train data copy
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
print('----------Train dataset copy completed')


# cat validation data copy
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# dog validation data copy
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
print('----------Validation dataset copy completed')


# cat test data copy
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# dog test data copy
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
print('----------Test dataset copy completed')

# 복사가 잘 되었는지 확인
print('Train cat images: ', len(os.listdir(train_cats_dir)))
print('Train dog images: ', len(os.listdir(train_dogs_dir)))
print('Valiadtion cat images: ', len(os.listdir(validation_cats_dir)))
print('Valiadtion dog images: ', len(os.listdir(validation_dogs_dir)))
print('Test cat images: ', len(os.listdir(test_cats_dir)))
print('Test cat images: ', len(os.listdir(test_dogs_dir)))

# Build network


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(150, 150, 3)))  # 이미지가 컬러라 마지막 3, 흑백이면 1
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data Preprocessing
# 이미지가 제가각이기 때문에 동일하게 맞출 필요가 있음

# Image scaling

# 1대255비율 사이즈를 150으로 잡아놔서 150으로 조정됨
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('Batch Data Size: ', data_batch.shape)
    print('Batch Label Size: ', labels_batch.shape)
    break  # 잘됐는지 확인 굳이 볼 필요 없으니 break ,

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=test_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_1.0.h5')  # 모델 저장 keras에서 지원함

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)


plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()  # 오버피팅 됨(학습 정확도는 증가, 검증 정확도는 제자리)
# 학습데이터를 늘리면 해결,
# 사진을 늘리는 방법- 사진을 회전, 반전 시킴, 이미지 증식

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

datagen = ImageDataGenerator(
    rotation_range=40,  # 40도 회전
    width_shift_range=0.2,  # 좌우로 조금씩 움직여봄 0.2 = 20%
    height_shift_range=0.2,  # 상하
    shear_range=0.2,  # 이미지의 기울기(사다리꼴 느낌)
    zoom_range=0.2,  # 회전
    horizontal_flip=True,  # 좌우반전
    fill_mode='nearest'
)


fnames = sorted(os.path.join(train_cats_dir, fname)
                for fname in os.listdir(train_cats_dir))

img_path = fnames[4]

# from tensorflow.keras.utils import load_img, img_to_array 사용하기 버전 달라서 에러나는 듯
# img = image.load_img(img_path, target_size=(150, 150)) 오류나서 대체
img = tf.keras.utils.load_img(img_path, target_size=(150, 150))
# x = image.img_to_array(img)  # 이미지 x 좌표
x = tf.keras.utils.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(tf.keras.utils.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()


#이미지를 회전시킨 다음 학습
train_datagen = ImageDataGenerator(    
    rescale=1./255,    
    rotation_range=40,    
    width_shift_range=0.2,    
    height_shift_range=0.2,    
    shear_range=0.2,    
    zoom_range=0.2,    
    horizontal_flip=True,)

# 학습은 오류남 이유찾기 오류 steps per epochs
# 검증 데이터는 증식되어서는 안 됩니다!
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(        
    # 타깃 디렉터리        
    train_dir,       
    # 모든 이미지를 150 × 150 크기로 바꿉니다        
    target_size=(150, 150),        
    batch_size=32,        
    # binary_crossentropy 손실을 사용하기 때문에 이진 레이블을 만들어야 합니다        
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(        
    validation_dir,        
    target_size=(150, 150),        
    batch_size=32,        
    class_mode='binary')

history = model.fit_generator(      
    train_generator,      
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)


plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''
for data_batch,labels_batch in train_generator:
    print('Batch Data Size', data_batch.shape)
    print('Battest_generatorel size')
'''