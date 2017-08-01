import numpy as np
import matplotlib.pyplot as plt

def AND(x, y):
  org = np.array([x, y])
  b = -0.7
  w = np.array([0.5, 0.5])
  print(w)
  print(w * x)
  return 0 < np.sum(w * x) + b

print(AND(1.0, 1.0))
