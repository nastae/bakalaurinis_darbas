# Lenet konvoliucinis neuroninis tinklas

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid', input_shape=(28, 28, 3)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='sigmoid'))
model.add(layers.Dense(units=84, activation='sigmoid'))
model.add(layers.Dense(units=7, activation='softmax'))

# VGG16, pritaikytas 28 x 28 vaizdams, konvoliucinis neuroninis tinklas

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 3), padding='same', activation='relu'))

model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(layers.MaxPooling2D())
model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='sigmoid'))
