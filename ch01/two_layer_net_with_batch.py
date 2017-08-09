import sys,os
import numpy as np
sys.path.append(os.pardir)
from common.functions import *
from common.functions import cross_entropy_error
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,
            weight_init_std=0.01):
        # 重みの初期化
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b2': np.zeros(output_size) }

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        return softmax(a2)

    def loss(self, x, t):
        return cross_entropy_error(self.predict(x), t)

    def accuracy(self, x, t):
        y = self.predict(x)
        # yとtの最大認識度のもののindex配列を出す(最大値は複数存在する可能性がある)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return  np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {'W1' : numerical_gradient(loss_W, self.params['W1']),
                 'b1' : numerical_gradient(loss_W, self.params['b1']),
                 'W2' : numerical_gradient(loss_W, self.params['W2']),
                 'b2' : numerical_gradient(loss_W, self.params['b1'])}
        return grads

#net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
#print(net.params['W1'].shape)
#print(net.params['b1'].shape)
#print(net.params['W2'].shape)
#print(net.params['b2'].shape)
#
#x = np.random.rand(100, 784)
#y = net.predict(x)
#
#x = np.random.rand(100, 784)
#t = np.random.rand(100, 10)
#
#grads = net.numerical_gradient(x, t) 
#print(grads['W1'].shape)
#print(grads['b1'].shape)
#print(grads['W2'].shape)
#print(grads['b2'].shape)

# (訓練画像、訓練ラベル), (テスト画像、テストラベル)
# >>> x_train.shape
#     (60000, 784)
#     60000件の、784(28x28画像をたぶんflatternしたもの)画像
# >>> t_train.shape
#     (60000, )
#     60000件の正解ラベル(t_train[0] = 1, t_train[1] = 5などが入っている)
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)

#[[ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# ..., 
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]]
#[5 0 4 ..., 5 6 8]

