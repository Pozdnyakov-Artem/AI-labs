import numpy as np

def kaiming(x, in_features):
    return x*np.sqrt(2/in_features)

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.W = kaiming(np.random.randn(in_features, out_features), in_features)
        self.b = np.zeros(out_features)
        self.bias = bias
        self.cache_x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.cache_x = x.copy()
        if self.bias:
            return np.matmul(x, self.W)+self.b
        return np.matmul(x, self.W)

    def backward(self, dout):
        x = self.cache_x
        self.dW[:] = np.matmul(x.transpose(), dout)
        dx = np.matmul(dout,self.W.transpose())
        if self.bias:
            self.db[:] = dout.sum(axis=0)
        return dx

    def parameters(self):
        yield self.W, self.dW
        yield self.b, self.db