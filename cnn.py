import keras
import keras.layers as layers
import cv2
import pandas as pd
import numpy as np
import os


datapath = './cat_dog_datasets/CatDog/'
model = keras.Sequential([
    layers.Input(shape=(128, 128, 1)),
    layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])


def image_process(path):
    image = cv2.imread(path)
    # cv2.imshow('1', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    return image


label = [0, 1]
X = []
y = []
for f, l in zip(os.listdir(datapath), label):
    print(f)
    for file in os.listdir(os.path.join(datapath, f)):
        X.append(image_process(os.path.join(datapath, f, file)))
        y.append(l)

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor='val_loss',
                         patience=20)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

model.fit(train_data,
          train_label,
          batch_size=64,
          epochs=10,
          validation_data=(test_data, test_label),
          shuffle=True,
          callbacks=[callback])




