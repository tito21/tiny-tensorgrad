
import numpy as np
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
        return Tensor(self.data[slice])

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, prev=(self, other), op="+")

        def _backward():
            self.grad += sum_if_need(self.grad.shape, out.grad)
            other.grad += sum_if_need(other.grad.shape, out.grad)

        out._backward_func = _backward

        return out

    def __radd__(self, other):
        return self + other

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

        assert self.data.shape[-1] == other.data.shape[0]
        out = Tensor(self.data @ other.data, prev=(self, other), op="@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward_func = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), prev=(self,), op="sum")

        def _backward():
            self.grad += np.expand_dims(out.grad, axis=axis if axis else ())

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
        dot.node(name=str(id(n)), label =f"{{ {n.label} | {n.data if n.shape == tuple() else n.shape} | grad: {np.linalg.norm(n.grad)}}}", shape='record')
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
