from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
NUM_CLASSES = 7

def model(input_shape, num_classes):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.0001),metrics=['accuracy'])

    return model

def fix_to_categorical(y_train):
    liste=[]
    for item in tf.keras.utils.to_categorical(y_train):
        item=np.delete(item,0)
        liste.append(item)
    y_train=np.array(liste)
    return y_train

