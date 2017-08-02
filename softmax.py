import numpy as np

def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a - c)
  sum_exp_a = np.sum(exp_a)
  return exp_a / sum_exp_a

a = np.array([0.3, 2.9, 4.0])
print(softmax(a))

print(np.sum(softmax(a)))
