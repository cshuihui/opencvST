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
train_data = MinMaxScaler().fit_transform(train_data)
train_label = df.iloc[:int(0.85*df.shape[0]),0].values
train_label = to_categorical(train_label, num_classes=10)
#测试集
test_data = df.iloc[int(0.85*df.shape[0]):,1:].values
test_label = df.iloc[int(0.85*df.shape[0]):,0].values
test_label = to_categorical(test_label, num_classes=10)
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)


# model = keras.saving.load_model("model.h5")
# print(model)


#现在准备搭模型 搭积木

# accuracy: 0.9051 - loss: 0.3076 - val_accuracy: 0.9034 - val_loss: 0.2967
# accuracy: 0.9808 - loss: 0.0753 - val_accuracy: 0.9521 - val_loss: 0.1656
# accuracy: 0.9686 - loss: 0.1085 - val_accuracy: 0.9585 - val_loss: 0.1375
# accuracy: 0.9228 - loss: 0.2517 - val_accuracy: 0.9185
# accuracy: 0.9577 - loss: 0.2932 - val_accuracy: 0.9143 - val_loss: 0.9136
# accuracy: 0.9705 - loss: 0.1264 - val_accuracy: 0.9496 - val_loss: 0.3395
# accuracy: 0.9632 - loss: 0.1621 - val_accuracy: 0.9510 - val_loss: 0.2918
# accuracy: 0.9317 - loss: 0.2215 - val_accuracy: 0.9319 - val_loss: 0.2146
model = keras.Sequential([
    keras.Input(shape=(784, )),
    layers.Dense(50, activation='sigmoid'),
    layers.Dense(50, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
])
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
model.load_weights("model.weights.h5")
# model = keras.saving.load_model("model.h5")
# score = model.evaluate(test_data, test_label)
# print(score)
print(test_data[0].shape) #(1,784)  (784,)
result = model.predict(test_data[0].reshape((1, 784)))
print(result, test_label[0])




# model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
# model.fit(train_data, train_label, batch_size=64, epochs=10, validation_split=0.1)
# model.save("model.h5")  #保存整个模型
# model.save_weights("model.weights.h5")


