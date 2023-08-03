import numpy as np
from sklearn.model_selection import train_test_split

n = 200_000
dim = 32
c = 3
x = np.random.randn(n, dim)
y = np.zeros((n, c))

for i in range(n):
    y[i, np.random.randint(0, c)] = 1

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

np.savetxt("x_train.txt", x_train)
np.savetxt("x_test.txt", x_test)
np.savetxt("y_train.txt", y_train)
np.savetxt("y_test.txt", y_test)
