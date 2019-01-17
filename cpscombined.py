# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 00:30:57 2018

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

# SKLEARN
from sklearn.model_selection import train_test_split
import vlc, easygui
import cv2



#face_cascade = cv2.CascadeClassifier('/home/vaibhavi/opencv/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
media = easygui.fileopenbox(title="Choose media to open")
player = vlc.MediaPlayer(media)
image = "WIN_20181107_16_55_35_Pro.jpg"
model = load_model('my_model.h5')


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



while True:
    choice = easygui.buttonbox(title="Smart Media Player",msg="Press Play to start",image=image,choices=["Play","Pause","Stop","New","Smart"])
    print(choice)
    if choice == "Play":
        player.play()
    elif choice == "Pause":
        player.pause()
    elif choice == "Stop":
        player.stop()
    elif choice == "New":
        media = easygui.fileopenbox(title="Choose media to open")
        player = vlc.MediaPlayer(media)
  
    elif choice == "Smart":
      cap = cv2.VideoCapture(0)
      player_actiqve = 0
      player_paused = 0

      player.play()
      player_active = 1 
      player_paused = 0
      
      while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    
        if len(faces) < 1 and (player_paused == 0 and player_active == 1):
            player.pause()
            player_active = 0
            player_paused = 1
            print ('Video Paused')


        elif len(faces)>0 and (player_active == 0 and player_paused == 1):
            player.play()
            player_active = 1 
            player_paused = 0
            print ('Video Playing')
      
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"press q to stop",(0,130), font, 1, (255,255,155))
        #cv2.imshow('frame', frame)
        
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
           player.stop()
           break
      cap.release()
      cv2.destroyAllWindows()
    else:
      break
