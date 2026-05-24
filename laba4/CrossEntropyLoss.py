import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.loss = None
        self.cache = None

    def forward(self, logits, labels):
        max_log = np.max(logits, axis=1, keepdims=True)
        exp_log = np.exp(logits - max_log)
        softmax = exp_log / np.sum(exp_log, axis=1, keepdims=True)

        N = len(labels)
        correct_logprobs = -np.log(softmax[range(N), labels] + 1e-8)

        self.loss = np.mean(correct_logprobs)
        self.cache = (softmax, labels, N)
        return self.loss

    def backward(self):
        softmax, labels, N = self.cache
        dscores = softmax.copy()
        dscores[range(N), labels] -= 1

        return dscores / N

    def __call__(self, x, y):
        return self.forward(x, y)