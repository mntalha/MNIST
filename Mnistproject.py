#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:25:01 2020

@author: talhakilic
"""

from keras.datasets import mnist
#load Mnist Dataset.
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
for i in range(15):
  plt.subplot(3,5,i+1)
  plt.tight_layout()
  plt.imshow(train_img[i])
  plt.title("Digit: {}".format(train_labels[i]))
  plt.xticks([])
  plt.yticks([])


#train and test images scaled and reshaped for model input
train_img = train_img.reshape((60000, 28, 28, 1))
train_img = train_img.astype('float32') / 255
test_img = test_img.reshape((10000, 28, 28, 1))
test_img = test_img.astype('float32') / 255


#one-hot encoder library ..
from keras.utils import to_categorical

#label values are transformed to 0 and 1 that is one hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#importing related library in keras
from keras import layers 
from keras import models

#create a model 
model = models.Sequential()

#apply 32 filter that has 3x3 pixel with activation of relu function.
#input must be 3 channel we have 28x28 image and 1 channel because of gray image
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
#select the best features or pixel that has high value.
model.add(layers.MaxPooling2D((2, 2)))

#make it this cycle 2 times more.
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#Flattening layer
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# summarize model.
model.summary()


#model compiling it may take a time. it will turn 5 times .
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_img, train_labels, epochs=5, batch_size=64)



