import numpy as np
import time
(n, m) = (2000, 2000)
(m, p) = (2000, 2000)
a = np.random.uniform(0, 1, (n, m))
b = np.random.uniform(0, 1, (m, p))
start = time.time()
c = a@b
duration = time.time() - start
print(f"Time elapsed in dot_matrix {a.shape} x {a.shape} = {c.shape} is: {duration} s")
