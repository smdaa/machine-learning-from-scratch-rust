import tensorflow as tf
import numpy as np

a = tf.constant([[1., 2., 1.], [.5, 1., .5]], dtype = tf.float32)
layer = tf.keras.layers.Softmax()


with tf.GradientTape(persistent=True) as tape:
    tape.watch(a)
    b = layer(a)

grad = tape.gradient(b, a)

print("a: \n", a)
print("b: \n", b)
print("grad: \n", grad)
