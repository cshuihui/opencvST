import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Bidirectional, LSTM
from keras.preprocessing import sequence
from keras.datasets import imdb

# 1. 参数设置
# 只保留训练数据中最常出现的前10,000个词，这是一个常见的词汇表大小设置[citation:1][citation:4]
VOCAB_SIZE = 10000
# 每个评论(序列)的最大长度，超过的截断，不足的补零，这是为了让输入数据形状一致
MAX_LEN = 200
# 将每个词映射成一个16维的稠密向量，这就是嵌入的维度[citation:4]
EMBEDDING_DIM = 128
# 训练时每次从数据集中取32个样本
BATCH_SIZE = 256
# 在整个训练集上迭代10轮
EPOCHS = 10

print("正在加载IMDb数据集...")
# 2. 加载并预处理数据
# num_words=VOCAB_SIZE 表示只保留最常出现的10000个词，其他词视为未知(用2表示)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print(f"原始训练集评论示例: {x_train[0][:10]} ...")
print(f"该评论的长度: {len(x_train[0])}")

# 3. 序列填充: 让所有评论长度统一为 MAX_LEN
# pad_sequences 会将超过 MAX_LEN 的序列截断，不足的用0在开头填充
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

print(f"\n处理后的输入数据形状: (样本数, 序列长度) = {x_train.shape}")

# 4. 构建模型架构 (Sequential 是线性堆叠的模型容器)
model = Sequential([
    # 嵌入层 (Embedding Layer):
    # 输入: (batch_size, MAX_LEN) 的整数索引矩阵
    # 输出: (batch_size, MAX_LEN, EMBEDDING_DIM) 的浮点数向量矩阵[citation:4]
    # 这里的参数数量 = VOCAB_SIZE * EMBEDDING_DIM = 10000 * 16 = 160,000
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),

    Bidirectional(LSTM(256, return_sequences=True)),
    # 双向 让模型同时从两个方向（从左到右和从右到左）处理数据

    # 全局平均池化层:
    # 作用: 在序列长度这个维度上求平均，将每个词16维的向量平均成一个16维的向量代表整句话[citation:5]
    # 输入: (batch_size, MAX_LEN, EMBEDDING_DIM)
    # 输出: (batch_size, EMBEDDING_DIM)
    GlobalAveragePooling1D(),

    # 全连接层 (输出层):
    # 输入: 经过池化后的16维向量
    # 输出: 1维 (0或1，代表正面或负面评价)
    Dense(1, activation='sigmoid')
])

# 5. 编译模型
# 优化器: adam 是一种常用的自适应学习率优化算法
# 损失函数: binary_crossentropy 是二分类问题的标准损失函数
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. 打印模型结构摘要，方便查看每一层的参数数量
model.summary()

# 7. 训练模型
print("\n开始训练...")
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.2)  # 从训练集中拿出20%作为验证集

# 8. 评估模型
print("\n在测试集上评估模型...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\n测试准确率: {test_acc:.4f}')