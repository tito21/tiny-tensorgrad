from typing import Sequence
import math

import numpy as np
from scipy.signal import correlate2d
from graphviz import Digraph

def sum_if_need(out_shape, in_data):
    try:
        np.broadcast_to(in_data, out_shape)
        return in_data
    except ValueError:
        return np.sum(in_data)

class Tensor:

    def __init__(self, data, label="", prev=(), op="") -> None:
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)

        self.label = label

        self._backward_func = lambda: None

        self._prev = prev
        self.op = op

    def __repr__(self):
        return f"Label: {self.label} data {self.data}, grad {self.grad} op {self.op}"

    def backward(self, gradient=None):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build_topo(c)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data) if gradient is None else gradient
        for n in reversed(topo):
            # print("tree grad", n.label, n.grad)
            n._backward_func()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(self.data.T, prev=(self,), op="T")

        def _backward():
            self.grad += out.grad.T

        out._backward_func = _backward

        return out

    def __getitem__(self, slice):
        out = Tensor(self.data[slice], prev=(self,), op="slice")

        def _backward():
            self.grad[slice] += out.grad

        out._backward_func = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, prev=(self, other), op="+")

        def _backward():
            self.grad += sum_if_need(self.grad.shape, out.grad)
            other.grad += sum_if_need(other.grad.shape, out.grad)

        out._backward_func = _backward

        return out

    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + self

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data, prev=(self, other), op="+")

        def _backward():
            self.grad += sum_if_need(self.grad.shape, out.grad)
            other.grad += sum_if_need(other.grad.shape, -out.grad)

        out._backward_func = _backward

        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, prev=(self, other), op="*")

        def _backward():
            self.grad += sum_if_need(self.grad.shape, other.data * out.grad)
            other.grad += sum_if_need(other.grad.shape, self.data * out.grad)

        out._backward_func = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1.0)


    def __rtruediv__(self, other):
        return other * (self**-1.0)

    def __pow__(self, k):
        if not isinstance(k, (int, float)):
            raise ValueError("Power only supported for float and integer values")

        out = Tensor(self.data**k, prev=(self,), op="pow")

        def _backward():
            self.grad += k * (self.data**(k-1)) * out.grad

        out._backward_func = _backward

        return out

    def __matmul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        assert self.data.shape[-1] == other.data.shape[0], "Incompatible shapes for matmul"
        out = Tensor(self.data @ other.data, prev=(self, other), op="@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward_func = _backward

        return out

    def convolve2d(self, f, stride=(1, 1), pad=((0, 0), (0, 0))):

        assert isinstance(f, Tensor), "Filter must be a tensor"
        assert f.data.ndim == 4, "Filter must have shape (channels_out, channels_in, kx, ky)"
        assert f.shape[1] == self.shape[1], "Filter must have the same number of channels as the tensor"
        assert self.data.ndim == 4, "To perform convolution the tensor must have dimensions (batch, channels_in, x, y)"

        def __stride_input(inputs, kx, ky):
                batch_size, channels, h, w = inputs.shape
                batch_stride, channel_stride, rows_stride, columns_stride = inputs.data.strides
                out_h = ((h - kx) // stride[0]) + 1
                out_w = ((w - ky) // stride[1]) + 1
                new_shape = (batch_size,
                            channels,
                            out_h,
                            out_w,
                            kx,
                            ky)
                new_strides = (batch_stride,
                            channel_stride,
                            stride[0] * rows_stride,
                            stride[1] * columns_stride,
                            rows_stride,
                            columns_stride)

                return np.lib.stride_tricks.as_strided(inputs, new_shape, new_strides)

        def correlate(inputs, filters):
            input_windows = __stride_input(inputs, filters.shape[2], filters.shape[3])
            output = np.einsum('bchwkt,fckt->bfhw', input_windows, filters, optimize=True)
            return output

        data = np.pad(self.data, ((0, 0), (0, 0), pad[0], pad[1]))
        out = correlate(data, f.data)
        out = Tensor(out, prev=(self, f), op="conv2d")

        def _backward():
            # out.grad.shape (bs, channels_out, ((h - kx) // stride[0]) + 1, ((w - ky) // stride[1]) + 1)
            _, _, h, w = self.shape
            _, _, kx, ky = f.shape
            pad_h = math.ceil(((h - 1)*stride[0] + kx - h) / 2)
            pad_w = math.ceil(((w - 1)*stride[1] + ky - w) / 2)
            # pad_h = (h - 1)*stride[0] + kx - h
            # pad_w = (w - 1)*stride[1] + ky - w

            # o = (h + 2*p - kx) // stride[0] + 1

            pad_h = h - out.grad.shape[2]
            pad_w = w - out.grad.shape[3]
            # print(pad_h, pad_w)
            padded_grad = np.pad(out.grad.copy(), ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))

            # print(out.grad.shape, self.data.shape, self.grad.shape, padded_grad.shape)
            # print(correlate(padded_grad, f.data[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)).shape)
            self.grad += correlate(padded_grad, f.data[:, :, ::-1, ::-1].transpose(1, 0, 2, 3))

            input_windows = __stride_input(data, out.grad.shape[2], out.grad.shape[3])
            f.grad += np.einsum('bchwkt,bfkt->fchw', input_windows, out.grad, optimize=True)

        out._backward_func = _backward

        return out

    def convolve2d_slow_single_channel(self, f):

        f = f if isinstance(f, Tensor) else Tensor(f)

        assert self.data.ndim == 3, "To perform convolution the tensor must have dimensions (batch, x, y)"

        out = np.stack([correlate2d(self.data[i, :, :], f.data, mode='valid') for i in range(self.shape[0])])
        out = Tensor(out, prev=(self, f), op="conv2d")

        def _backward():
            self.grad += np.stack([correlate2d(out.grad[i, :, :], f.data[::-1, ::-1], mode='full') for i in range(self.shape[0])])
            f.grad += np.sum(np.stack([correlate2d(self.data[i, :, :], out.grad[i, :, :], mode='valid') for i in range(self.shape[0])]), 0)

        out._backward_func = _backward

        return out

    def avg_pooling(self, kernel_size=(2, 2)):
        bs, c, y, x = self.shape
        ny = y // kernel_size[0]
        nx = x // kernel_size[1]
        data_pad = self.data[..., :ny*kernel_size[0], :nx*kernel_size[1]]
        out = data_pad.reshape(bs, c, ny, kernel_size[0], nx, kernel_size[1]).mean(axis=(3, 5))
        out = Tensor(out, prev=(self,), op="avg_pool")

        def _backward():
            g = np.repeat(np.repeat(out.grad, kernel_size[0], axis=-2), kernel_size[1], axis=-1)/np.prod(kernel_size)
            pad_y = y - ny*kernel_size[0] if ny*kernel_size[0] < y else 0
            pad_x = x - nx*kernel_size[1] if nx*kernel_size[1] < x else 0
            g = np.pad(g, ((0, 0), (0, 0), (0, pad_y), (0, pad_x)))
            # print(g.shape, out.grad.shape, self.data.shape, self.grad.shape)
            self.grad += g

        out._backward_func = _backward

        return out

    @staticmethod
    def stack(tensors, axis=0):
        out = Tensor(np.stack([t.data for t in tensors], axis), prev=tuple(tensors), op="stack")

        def _backward():
            for i, t in enumerate(tensors):
                t.grad += out.grad.take(i, axis=axis)

        out._backward_func = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), prev=(self,), op="sum")

        def _backward():
            self.grad += np.expand_dims(out.grad, axis=axis if axis else ())

        out._backward_func = _backward

        return out

    def reshape(self, new_shape):
        out = Tensor(np.reshape(self.data.copy(), new_shape), prev=(self,), op="reshape")

        def _backward():
            self.grad += np.reshape(out.grad.copy(), self.grad.shape)

        out._backward_func = _backward

        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), prev=(self,), op="tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward_func = _backward

        return out

    def relu(self):
        out = Tensor((self.data > 0) * self.data, prev=(self,), op="relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward_func = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), prev=(self,), op="exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward_func = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), prev=(self,), op="log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward_func = _backward

        return out

    def sigmoid(self):
        out = Tensor(1.0 / (1 + np.exp(-self.data)), prev=(self,), op="sigmoid")

        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward_func = _backward

        return out

    def softmax(self, axis=-1):
        def sm(x):
            exp_in = np.exp(x - x.max(axis, keepdims=True))
            denominator = np.sum(exp_in, axis, keepdims=True)
            return exp_in / denominator

        out = Tensor(sm(self.data), prev=(self,), op="softmax")

        def _backward():
            bs_grads = []
            for bs in range(self.shape[0]):
                x_grad = (np.diag(out.data[bs]) - np.outer(out.data[bs], out.data[bs]))
                bs_grads.append(x_grad @ out.grad[bs].T)
            self.grad += np.stack(bs_grads)

        out._backward_func = _backward

        return out

# Drawing utils
def trace(root: Tensor):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label =f"{{ {n.label} | {n.data if n.shape == tuple() else n.shape} | grad norm: {np.linalg.norm(n.grad)}}}", shape='record')
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
