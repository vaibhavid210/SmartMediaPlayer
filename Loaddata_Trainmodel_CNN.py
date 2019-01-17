# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:27:46 2018

@author: vishw
"""
import os
import theano
#os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras import optimizers
from keras.utils import np_utils
import keras
import h5py
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


from sklearn.model_selection import train_test_split


loaded_pixels = np.load('pixels_data.npy')
loaded_label = np.load('label.npy')
loaded_label = np.resize(loaded_label,32298)
loaded_pixels_normalized = loaded_pixels/255 
loaded_pixels_normalized = loaded_pixels_normalized.astype('float32')
#test = loaded_pixels[100,:]
#test = np.reshape(test,(48,48))
#plt.imshow(test)
#plt.imshow(test,cmap='gray')

train_data = [loaded_pixels_normalized,loaded_label]
(X,y) = (train_data[0],train_data[1])

batch_size = 64
# number of output classes
nb_classes = 7
# number of epochs to train
nb_epoch = 100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train = X_train.reshape(X_train.shape[0], 1, 48,48)
X_test = X_test.reshape(X_test.shape[0], 1, 48,48)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, (nb_conv, nb_conv),
                        border_mode='valid',
                        input_shape=(1,48, 48)))
convout1 = Activation('relu')
model.add(BatchNormalization())
model.add(convout1)
model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
convout2 = Activation('relu')
model.add(convout2)
model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv)))
convout3 = Activation('relu')
model.add(convout3)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_data=(X_test, Y_test))

model.save('hist.h5')





