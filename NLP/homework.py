import keras

max_features = 10000  # 只保留词频前20000的词


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


inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(max_features, 256)(inputs)
x = layers.SpatialDropout1D(0.3)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
x = layers.GlobalMaxPool1D()(x)  # 替代最后一个LSTM的return_sequences=False

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(46, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)
