#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:37:35 2019

@author: kiran allada
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Intilaizing the CNN
model = Sequential()
#convolution layer
model.add(Convolution2D(32,3,3,input_shape = (64,64,3)))
#Maxpooling layer()
model.add(MaxPooling2D(pool_size=(2,2)))
#flattening the neural network
model.add(Flatten())
#fully connected layer
model.add(Dense(output_dim = 128,activation = 'relu'))
#output layer
model.add(Dense(output_dim = 1,activation = 'sigmoid'))
#compiling the data
model.compile(optimizer= 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
#preprocessing the data
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'chest_xray/train',
        target_size=(64, 64),
        batch_size=32, 
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'chest_xray/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=800)
#Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('chest_xray/prediction/3.jpeg',target_size = (64,64))
test_image = image.img_to_array(test_image)
#add another dimension (predict accepts 4 dimensions)
test_image = np.expand_dims(test_image,axis = 0)
result = model.predict(test_image)

train_generator.class_indices
