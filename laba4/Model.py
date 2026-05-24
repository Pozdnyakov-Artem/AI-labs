import numpy as np

class Model:
    def __init__(self):
        self.layers=[]
        self.training = True

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

    def train(self):
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()