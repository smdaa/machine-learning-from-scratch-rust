import tensorflow as tf

y_true = [0, 1, 0, 0]
y_pred = [-18.6, 0.51, 2.94, -12.8]

y_true_tensor = tf.constant(y_true, dtype=tf.float32)
y_pred_tensor = tf.constant(y_pred, dtype=tf.float32)

bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

with tf.GradientTape() as tape:
    tape.watch(y_pred_tensor)
    loss_value = bce_loss(y_true_tensor, y_pred_tensor)

gradient = tape.gradient(loss_value, y_pred_tensor)

print("loss_value : ", loss_value.numpy())
print("gradient : ", gradient.numpy())
