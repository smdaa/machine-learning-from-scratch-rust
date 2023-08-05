import tensorflow as tf
import numpy as np

logits = [-3.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -3.0]
logits = np.array(logits).reshape((4, 2))
logits_tensor = tf.constant(logits, dtype=tf.float32)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(logits_tensor)
    layer = tf.keras.layers.ReLU()
    output = layer(logits_tensor)

gradients = tape.gradient(output, logits_tensor)

print("Input logits: \n", logits)
print("ReLU output: \n", output.numpy())
print("Gradients of ReLU with respect to logits: \n", gradients.numpy())
