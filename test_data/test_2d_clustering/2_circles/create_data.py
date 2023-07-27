#!/usr/bin/env python3

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def save_x_txt(x, s):
    f = open(s+".txt", "w+")
    for i in range(x.shape[0]):
        s = ""
        for j in range(x.shape[1]):
            s += f"{x[i, j]:.2} "
        s += "\n"
        f.write(s)
    f.close()

def save_y_txt(y, s):
    f = open(s+".txt", "w+")
    for i in range(y.shape[0]):
        s = f"{y[i]}\n"
        f.write(s)
    f.close()


x, y = make_circles(n_samples=10000, factor=0.1, noise=0.1, random_state=0)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

save_x_txt(x_train, "x_train")
save_x_txt(x_test, "x_test")
save_y_txt(y_train, "y_train")
save_y_txt(y_test, "y_test")