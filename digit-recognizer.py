# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import math  
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
def plotDigit(img):

    plt.imshow(img,cmap=plt.cm.binary)
    plt.show()
    
def reshape(df):
    return df.values.reshape(df.shape[0],int(math.sqrt(df.shape[1])),int(math.sqrt(df.shape[1])))
    
train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

x_train, x_test ,y_train, y_test =  train_test_split(train.iloc[:,1:], train["label"], test_size=0.001)




x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

test = tf.keras.utils.normalize(test, axis=1)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, epochs=500 ,batch_size =1000  )  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy

prd= model.predict_classes(test)

df_subm = pd.read_csv('sample_submission.csv')
df_subm['Label'] =prd
df_subm.to_csv('submission.csv', index=False)