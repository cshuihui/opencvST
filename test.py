import tensorflow as tf
import time

# 创建两个大矩阵
matrix_size = 5000
a = tf.random.normal((matrix_size, matrix_size))
b = tf.random.normal((matrix_size, matrix_size))

print("开始矩阵乘法运算...")
start_time = time.time()

# 执行矩阵乘法
c = tf.matmul(a, b)

# 强制立即执行（在Eager模式下）
if tf.executing_eagerly():
    c.numpy()  # 这会触发计算

end_time = time.time()
print(f"运算耗时: {end_time - start_time:.2f} 秒")

# 查看运算设备
print(f"\n运算设备: {c.device}")