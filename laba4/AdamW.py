import numpy as np

class AdamW:
    def __init__(self, params_iter, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params_iter)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {id(p): {
                'm': np.zeros_like(p),
                'v': np.zeros_like(p)
            } for p, _ in self.params}
        self.timestep = 0

    def step(self):
        self.timestep += 1
        for p, g in self.params:
            if g is None: continue
            pid = id(p)
            self.state[pid]['m'] = self.betas[0] * (self.state[pid]['m']) + (1-self.betas[0]) * g
            self.state[pid]['v'] = self.betas[1]*self.state[pid]['v'] + (1-self.betas[1])*g**2
            mt = self.state[pid]['m']/(1-self.betas[0]**self.timestep)
            v = self.state[pid]['v']/(1-self.betas[1]**self.timestep)
            p -= self.lr * (mt/(np.sqrt(v)+self.eps) + self.weight_decay * p)

    def zero_grad(self):
        for p, g in self.params:
            g.fill(0)