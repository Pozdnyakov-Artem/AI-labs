import numpy as np


class Dropout:
    def __init__(self, p=0.3):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        return x * self.mask / (1 - self.p)

    def backward(self, dout):
        if not self.training or self.p == 0:
            return dout
        return dout * self.mask / (1 - self.p)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True