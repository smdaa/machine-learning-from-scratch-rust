import tensorflow as tf
import numpy as np

a = tf.constant([[1.0, 2.0, 1.0], [0.5, 1.0, 0.5]], dtype=tf.float32)

custom_weights = np.array(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [8.0, 9.0, 10.0, 11.0]],
)
custom_bias = np.array([1.0, 2.0, 3.0, 4.0])
layer = tf.keras.layers.Dense(
    units=4,
    kernel_initializer=tf.constant_initializer(custom_weights),
    bias_initializer=tf.constant_initializer(custom_bias),
)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(a)
    b = layer(a)

grad = tape.gradient(b, a)
dw = tape.gradient(b, layer.kernel)
db = tape.gradient(b, layer.bias) 

print("a: \n", a)
print("b: \n", b)
print("grad: \n", grad)
print("dw: \n", dw)
print("db: \n", db)
