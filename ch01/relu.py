import numpy as np
class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        # マスク部分を0にする
        # [[ 2.  -0.6]
        #  [ 1.  -0.5]]
        #   => 
        #    [[ 2.  0.]
        #     [ 1.  0.]]
        out[self.mask] = 0
        return out

x = np.array([[2.0,-0.6], [1.0,-0.5]])
relu_layer = ReluLayer()
relu_layer.forward(x)
