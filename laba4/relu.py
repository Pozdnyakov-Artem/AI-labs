import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x > 0)
        return np.where(self.mask, x, 0)

    def backward(self,dout):
        return dout*self.mask