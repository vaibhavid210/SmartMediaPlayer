# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:36:06 2018

@author: vishw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 03:48:22 2018

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
import cv2

model = load_model('my_model.h5')



cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def number_to_strings(a):
           switcher = {
                0: "Angry",
                1: "Disgust",
                2: "Fear",
                3: "Happy",
                4: "Sad",
                5: "Surprise",
                6: "Neutral",
             }
           return (switcher.get(a, "Invalid"))

while(True):    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    
    for (x,y,w,h) in faces:
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0 , 0, 255), 2)
       out = gray[y:y+h, x:x+w]
       font = cv2.FONT_HERSHEY_SIMPLEX
       out = cv2.resize(out, (48, 48))       
       first = out
       out_normalized = out/255
       Xs = np.expand_dims(out_normalized, axis=0)
       Xs = Xs.reshape(1,1,48,48)       
       a=model.predict_classes(Xs)               
       a = int(a)
       emotion = (number_to_strings(a))
       cv2.putText(frame,emotion,(y+h,x-140), font, 1,(0,0,255),2,cv2.LINE_AA)
       cv2.imshow('frame', frame)
       cv2.waitKey(300)
       #print(number_to_strings(a))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

cap.release()

cv2.destroyAllWindows()





