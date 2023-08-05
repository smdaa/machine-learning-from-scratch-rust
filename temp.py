import tensorflow as tf

y_true = [[0, 1, 0], [0, 0, 1]]
logits = [[-18.6, 0.51, 2.94], [-12.8, 0.40, 3.95]]

y_true_tensor = tf.constant(y_true, dtype=tf.float32)
logits_tensor = tf.constant(logits, dtype=tf.float32)
y_pred_tensor =  tf.nn.softmax(logits)

bce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

with tf.GradientTape() as tape:
    tape.watch(logits_tensor)
    loss_value = bce_loss(y_true_tensor, logits_tensor)

gradient = tape.gradient(loss_value, logits_tensor)

print("loss_value : \n", loss_value.numpy())
print("gradient : \n", gradient.numpy())
print("activation : \n", y_pred_tensor.numpy())
