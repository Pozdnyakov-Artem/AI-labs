import numpy as np

class Model:
    def __init__(self):
        self.layers=[]

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                yield from layer.parameters()

    def __call__(self, x):
        return self.forward(x)