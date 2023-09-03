import numpy as np

def Householder(x, i):
    alpha = -np.sign(x[i]) * np.linalg.norm(x)
    e = np.zeros(len(x)); e[i] = 1.0
    
    v = (x - alpha * e)
    w = v / np.linalg.norm(v)
    P = np.eye(len(x)) - 2 * np.outer(w, w.T)
    
    return P

