import numpy as np

x_train = np.load("./train-images.npy")
y_train = np.load("./train-labels.npy")
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
y_train_onehot = np.zeros((y_train.size, 10))
for i in range(y_train.size):
    y_train_onehot[i, y_train[i]] = 1.0
x_train = x_train / 255
print(x_train.shape)
print(y_train_onehot.shape)
np.savetxt("x_train.txt", x_train)
np.savetxt("y_train.txt", y_train_onehot)

x_test = np.load("./t10k-images.npy")
y_test = np.load("./t10k-labels.npy")
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
y_test_onehot = np.zeros((y_test.size, 10))
for i in range(y_test.size):
    y_test_onehot[i, y_test[i]] = 1.0
x_test = x_test / 255
print(x_test.shape)
print(y_test_onehot.shape)
np.savetxt("x_test.txt", x_test)
np.savetxt("y_test.txt", y_test_onehot)

