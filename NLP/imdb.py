import keras

max_features = 20000  # 只保留词频前20000的词


(x_train, y_train), (x_val, y_val) = keras.datasets.reuters.load_data(
    num_words=max_features
)

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
print(x_train[0])

maxlen = 200  # 每条评论只看前200个词

#填充
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

print(x_train.shape)

from keras import layers

max_features = 20000
maxlen = 200

inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(max_features, 128)(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=2,
    validation_data=(x_val, y_val)
)
