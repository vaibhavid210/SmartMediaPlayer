# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 18:17:39 2018

@author: vishw
"""





import numpy as np
import pandas as pd



df = pd.read_csv('fer2013.csv')
label = df.emotion
image_data = df.pixels
label = np.array(label)
image_data = np.array(image_data)
#image_data = image_data.astype('float32')
#image_data[0] = image_data[0].split(" ")
#print(df.str.split(expand=True))
fin_image_data = np.empty([32298,2304])
for i in range(32298):
  df_temp = pd.Series((image_data[i]))
  fin_image_data[i,:] = df_temp.str.split(expand=True)
  #fin_image_data[i,:] = df_temp  
print(fin_image_data.shape)
np.save('pixels_data.npy',fin_image_data)
np.save('label.npy',label)
loaded_pixels = np.load('pixels_data.npy')

  
  
  
  
  
  
  
  
  

