from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

n_samples = 10_000
x, y = make_circles(n_samples, factor=0.3, noise=0.05, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

np.savetxt("x_train.txt", x_train)
np.savetxt("x_test.txt", x_test)
np.savetxt("y_train.txt", y_train)
np.savetxt("y_test.txt", y_test)
