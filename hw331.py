import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import keras.layers as layers
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping

data = pd.read_csv('winequality-white.csv', sep=';')
print(data.shape)

data.iloc[:, 11] = data.iloc[:, 11] - 3  # 相当于把不是从0开始的标签映射成从0开始的

train_data, test_data, train_label, test_label = train_test_split(data.iloc[:, :11], data.iloc[:, 11],
                                                                  train_size=0.9, random_state=42, shuffle=True)

classes = np.unique(data.iloc[:, 11])
print(classes)

scale = StandardScaler()
train_data = scale.fit_transform(train_data)
test_data = scale.fit_transform(test_data)

model = keras.Sequential([
    keras.Input(shape=(11,)),

    layers.Dense(128,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.01),
    layers.Dropout(0.1),

    layers.Dense(64,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.01),
    layers.Dropout(0.1),

    layers.Dense(64,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.01),
    layers.Dropout(0.1),

    layers.Dense(64,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.01),
    layers.Dropout(0.1),

    layers.Dense(32,
                 kernel_regularizer=keras.regularizers.l2(0.001)),  # l2正则化
    layers.BatchNormalization(),  # 每个batch后重新标准化
    layers.LeakyReLU(alpha=0.01),
    layers.Dropout(0.1),

    layers.Dense(16,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.01),
    layers.Dropout(0.1),

    layers.Dense(16,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.01),


    layers.Dense(len(classes), activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=0.0005),
              metrics=["accuracy"])
# categorical_crossentropy
# one-hot 转换成n个种类的标签相对应的标签上为1其余为零 [0,0,1,0]
# 要求one-hot类 其索引必须从0开始

early_stop = EarlyStopping(
    monitor='val_accuracy',  # 监控
    patience=200,  # n轮不提升停止
    restore_best_weights=True  # 恢复最好模型
)
model.fit(train_data,
          train_label,
          batch_size=64,
          epochs=1000,
          validation_data=(test_data, test_label),
          callbacks=[early_stop],
          shuffle=True)

loss, acc = model.evaluate(test_data, test_label)
print("Test Loss:", loss)
print("Test Accuracy:", acc)