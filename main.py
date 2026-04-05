import keras
import keras.layers as layers
import pandas as pd
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#加载train.csv里面的数据 到 df
df = pd.read_csv("train.csv")
pic1_data = df.iloc[15, 1:].values
print(pic1_data.shape)
# pic1_data = pic1_data.reshape((28, 28))
# pic1_data = pic1_data.astype(np.uint8)
# cv2.imshow("sfdsf", pic1_data)
# cv2.waitKey(0)

#训练集出来了
train_data = df.iloc[:int(0.85*df.shape[0]),1:].values
train_label = df.iloc[:int(0.85*df.shape[0]),0].values
train_label = to_categorical(train_label, num_classes=10)
#测试集
test_data = df.iloc[int(0.85*df.shape[0]):,1:].values
test_label = df.iloc[int(0.85*df.shape[0]):,0].values
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

#现在准备搭模型 搭积木
model = keras.Sequential([
    keras.Input(shape=(784, )),
    layers.Rescaling(1./255),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10个分类概率总和为1 概率分布 sigmoid 用于二分类
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])  # loss那里是损失函数 适用于多分类
model.fit(train_data, train_label, batch_size=64, epochs=10, validation_split=0.3)
