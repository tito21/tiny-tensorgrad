from typing import List, Sequence, Union

import numpy as np

from engine import Tensor

def uniform(shape, min=-1, max=1):
    return (max - min)*np.random.rand(*shape) + min

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

        self.weights = Tensor(uniform((features_out, features_in), -np.sqrt(1/features_in), np.sqrt(1/features_in)), label=name+" Weight")
        if use_bias:
            self.bias = Tensor(uniform((features_out,), -np.sqrt(1/features_in), np.sqrt(1/features_in)), label=name+" bias")

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

        sqrt_k = np.sqrt(1/(channels_in)*np.prod(kernel_size))
        self.weights = Tensor(uniform((channels_out, channels_in, *kernel_size), -sqrt_k, sqrt_k), label=name+" Weight")
        if use_bias:
            self.bias = Tensor(uniform((1, channels_out, 1, 1), -sqrt_k, sqrt_k), label=name+" bias")

    def __call__(self, x: Tensor) -> Tensor:

        out = x.convolve2d(self.weights)
        if self.use_bias:
            out += self.bias

        return out

    def parameters(self):
        return [self.weights, self.bias]

class AvgPooling(Module):
    def __init__(self, kernel_size=(3, 3), name=""):
        self.name = name
        self.kernel_size = kernel_size

    def __call__(self, x: Tensor):
        assert x.data.ndim == 4, "Input must have 4 dimensions (batch, channels, x, y)"
        return x.avg_pooling(self.kernel_size)

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
