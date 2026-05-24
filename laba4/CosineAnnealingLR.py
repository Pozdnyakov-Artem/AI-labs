import numpy as np

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=1e-6, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        self.lr_max = self.optimizer.lr
        self.t = last_epoch

    def step(self):
        self.t += 1
        self.optimizer.lr = self.eta_min + 0.5 * (self.lr_max - self.eta_min) * (1 + np.cos(np.pi*self.t / self.T_max))
