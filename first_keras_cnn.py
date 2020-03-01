# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:19:08 2020

@author: Andrew
"""


from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#Getting and seperating data into train and test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Reshaping
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#Changing into categorical (vector with a 1 on it)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Buidling model, two convolutional layers using relu, softmax so it can make predictions on probabilities
model = Sequential()
model.add(Conv2D(64,kernel_size=3,activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compliling 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)


