import time
import numpy as np
import tensorflow as tf


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    return model


# load data
x_train = np.loadtxt("x_train.txt")
x_test = np.loadtxt("x_test.txt")
y_train = np.loadtxt("y_train.txt")
y_test = np.loadtxt("y_test.txt")

print("x_train.shape :", x_train.shape)
print("x_test.shape :", x_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape :", y_test.shape)

# load model
model = create_model()
learning_rate = 0.01
momentum = 0.0
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# train model
start = time.time()
epochs = 1000
batch_size = 32
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# test model
_, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy : %.3f" % (accuracy * 100.0))

end = time.time()
print('elapsed time', end - start)
