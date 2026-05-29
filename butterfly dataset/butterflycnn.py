import keras
import pandas as pd
import cv2
import numpy as np
import os
# print(os.path.dirname(os.path.abspath(__file__)))
# 在docker里最好使用绝对路径避免错误
script_dir = os.path.dirname(os.path.abspath(__file__))


def image_process(path):
    image = cv2.imread(os.path.join(script_dir, path))
    # 在docker里使用opencv 为了不麻烦不要使用需要调用图形化界面的功能 imshow之类 waitkey 也是
    # cv2.imshow('1', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image


train_data = pd.read_excel(os.path.join(script_dir, 'Training_set.xlsx'))
test_data = pd.read_excel(os.path.join(script_dir, 'Test_set.xlsx'))
train_img = []
test_img = []

for path in train_data['filename']:
    train_img.append(image_process(os.path.join(script_dir, 'train', path)))

for path in test_data['filename']:
    test_img.append(image_process(os.path.join(script_dir, 'test', path)))

train_img, test_img = np.array(train_img), np.array(test_img)

from sklearn.preprocessing import LabelEncoder

# 如果标签是字符串（如 'Monarch', 'Swallowtail' 等）
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['label'])
import pickle

with open(os.path.join(script_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
test_labels = label_encoder.transform(test_data['label'])


from keras.utils import to_categorical

# 然后转换为 one-hot
# 假如一个数据的标签对应数字是2  那么转换后输入给模型的就是[0, 0, 1]
num_classes = len(label_encoder.classes_)
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


from keras.applications.resnet50 import ResNet50
base_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
#                                                                     去掉模型顶部的全连接分类层

for layer in base_model.layers[:-10]:
    layer.trainable = False

from keras import layers
model = keras.Sequential([
    base_model,
    layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.GlobalAveragePooling2D(),  # 2D -> 1D
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(75, activation='softmax')
])

from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor='val_loss',
                         patience=3,
                         )


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_img,
          train_labels,
          batch_size=64,
          epochs=10,
          validation_data=(test_img, test_labels),
          shuffle=True,
          callbacks=[callback])

model.save_weights(os.path.join(script_dir, 'butterfly.keras'))
