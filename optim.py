from typing import Sequence

import numpy as np

from engine import Tensor

class Optim:
    def __init__(self, parameters: Sequence[Tensor]) -> None:
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class SGD(Optim):
    def __init__(self, parameters: Sequence[Tensor], lr: float, momentum=0.0):
        super().__init__(parameters)

        self.lr = lr
        self.momentum = momentum
        self.b = [np.zeros_like(p.grad) for p in self.parameters]

    def step(self):
        for b, p in zip(self.b, self.parameters):
            # print(p.label, np.linalg.norm(p.grad))
            b = self.momentum * b + p.grad
            p.data -= self.lr*(p.grad + self.momentum*b)

class Adam(Optim):
    def __init__(self, parameters: Sequence[Tensor], lr: float, betas=(0.9, 0.999), eps=1e-8) -> None:
        super().__init__(parameters)

        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.m = [np.zeros_like(p.grad) for p in self.parameters]
        self.v = [np.zeros_like(p.grad) for p in self.parameters]

        self.t = 1

    def step(self):
        for m, v, p in zip(self.m, self.v, self.parameters):

            m = self.betas[0]*m + (1 - self.betas[0])*p.grad
            v = self.betas[1]*v + (1 - self.betas[1])*p.grad**2

            m_hat = m / (1 - self.betas[0]**self.t)
            v_hat = v / (1 - self.betas[1]**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.t += 1
