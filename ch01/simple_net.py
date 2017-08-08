import sys,os
import numpy as np
sys.path.append(os.pardir)
from common.functions import *
from common.functions import cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y,t)

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print("predict")
print(p)
print("max")
print(np.argmax(p))

t = np.array([0, 0, 1])
net.loss(x, t)

def f(W):
    return net.loss(x, t)
dw = numerical_gradient(f, net.W)
print(dw)
#class SimpleNet
#
#    def __init__(self, input_size, hidden_size, output_size,
#            weight_init_std=0.01)
#        # 重みの初期化
#        self.params = {}
#        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#        self.params['b1'] = np.zeros(hidden_size)
#        self.params['W2'] = weight_init_std * np.random.randn(hidden_size * output_size)
#        self.params['b2'] = np.zeros(output_size)
