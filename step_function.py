import numpy as np

def step_function(x):
  y = x > 0
  print(y)
  return y.astype(np.int)

res = step_function(np.array([0,1,3]))
print(res)
