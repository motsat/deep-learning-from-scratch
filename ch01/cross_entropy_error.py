import numpy as np

def cross_entropy_error(y, t):
    dalta = 1e-7
    return -np.sum(t * np.log(y + dalta))

t = np.array([0, 0, 1, 0, 0, 0, 0,  0, 0, 0 ])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0,  0.1, 0, 0 ])
print(cross_entropy_error(np.array(y), np.array(t)))
