import numpy as np

from engine import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for m in self.modules:
            x = m(x)
        return x

    def parameters(self):
        return sum((m.parameters() for m in self.modules), start=[])

class Linear(Module):
    def __init__(self, features_in, features_out, use_bias=True, name=""):

        self.features_in = features_in
        self.features_out = features_out
        self.use_bias = use_bias
        self.name = name

        self.weights = Tensor(2*np.random.rand(features_out, features_in) - 1, label=name+" Weight")
        if use_bias:
            self.bias = Tensor(2*np.random.rand(features_out) - 1, label=name+" bias")

    def __call__(self, x: Tensor):
        out = x @ self.weights.T
        if self.use_bias:
            out = out + self.bias

        return out

    def parameters(self):
        return [self.weights, self.bias]

class Tanh(Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x.tanh()

class ReLU(Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x.relu()