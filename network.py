import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

X = np.array([ 1.0, 0.5])
W = np.array([ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ])
B = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X,W) + B
print(A1)

Z1 = sigmoid(A1)
print(Z1)
