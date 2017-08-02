import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def identity_function(x):
  return x

def init_newtwork():
  network = {}
  network['W1'] = np.array([ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([ [0.1, 0.4], [0.2, 0.5], [0.3, 0.6] ])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([ [0.1, 0.3], [0.2, 0.4] ])
  network['b3'] = np.array([0.1, 0.2])

  return network

def forward(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']
  
  A1 = np.dot(x, W1) + b1
  Z1 = sigmoid(A1)
  A2 = np.dot(Z1, W2) + b2
  Z2 = sigmoid(A2)
  A3 = np.dot(Z2, W3) + b3

  return identity_function(A3)

x = np.array([ 1.0, 0.5])
print(forward(init_newtwork(), x))

