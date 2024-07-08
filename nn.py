from typing import List, Sequence, Union

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

    def __call__(self, x: Tensor):
        for m in self.modules:
            x = m(x)
        return x

    def parameters(self):
        return sum((m.parameters() for m in self.modules), start=[])

class Linear(Module):
    def __init__(self, features_in: int, features_out: int, use_bias=True, name=""):

        self.features_in = features_in
        self.features_out = features_out
        self.use_bias = use_bias
        self.name = name

        self.weights = Tensor(2*np.random.rand(features_out, features_in) - 1, label=name+" Weight")
        if use_bias:
            self.bias = Tensor(2*np.random.rand(features_out) - 1, label=name+" bias")

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.weights.T
        if self.use_bias:
            out = out + self.bias

        return out

    def parameters(self):
        return [self.weights, self.bias]

class Conv2d(Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size=(3, 3), use_bias=True, name=""):

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.use_bias = use_bias
        self.name = name

        self.weights = Tensor(2*np.random.rand(channels_out, channels_in, *kernel_size) - 1, label=name+" Weight")
        if use_bias:
            self.bias = Tensor(2*np.random.rand(1, channels_out, 1, 1) - 1, label=name+" bias")

    def __call__(self, x: Tensor) -> Tensor:

        out = x.convolve2d(self.weights)
        if self.use_bias:
            out += self.bias

        return out

    def parameters(self):
        return [self.weights, self.bias]

class Flatten(Module):
    def __init__(self) -> None:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x.reshape((x.shape[0], -1))

class Tanh(Module):
    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()

class ReLU(Module):
    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Module):
    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Softmax(Module):
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        return x.softmax(self.axis)
