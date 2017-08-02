import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def identity_function(x):
  return x

X = np.array([ 1.0, 0.5])
W1 = np.array([ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
print(A1)

Z1 = sigmoid(A1)
print(Z1)

W2 = np.array([ [0.1, 0.4], [0.2, 0.5], [0.3, 0.6] ])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
print(A2)
Z2 = identity_function(A2)
print(Z2)
