# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:54:37 2020

@author: acer
"""

#Importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Intialize the Model
model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=128,init='uniform',activation='relu'))
model.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
x_train=train_datagen.flow_from_directory(r'D:/Project/Dataset/train_set',target_size=(64,64),batch_size=32,class_mode='binary')
x_test=test_datagen.flow_from_directory(r'D:/Project/Dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')
print(x_train.class_indices)
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])

#Fit the model
model.fit_generator(x_train,steps_per_epoch=35,epochs=10,validation_data=x_test,validation_steps=5)

#Save the model
model.save("cnn.h5")






# Predicting the Model
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
model=load_model("cnn.h5")
img=image.load_img(r"C:\Users\acer\Desktop\cancer.png",target_size=(64,64))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=model.predict_classes(x)
pred