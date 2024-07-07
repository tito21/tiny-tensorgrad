import unittest

import numpy as np
import torch
import torch.nn.functional as F

from engine import Tensor

SHAPE = (4, 8)

class TestOp(unittest.TestCase):

    def test_add(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a + b

        a = torch.from_numpy(a.data)
        b = torch.from_numpy(b.data)

        out_torch = a + b
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_sub(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a - b

        a = torch.from_numpy(a.data)
        b = torch.from_numpy(b.data)

        out_torch = a - b
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_mul(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a * b

        a = torch.from_numpy(a.data)
        b = torch.from_numpy(b.data)

        out_torch = a * b
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_div(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a / b

        a = torch.from_numpy(a.data)
        b = torch.from_numpy(b.data)

        out_torch = a / b
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_pow(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = np.random.rand()

        out = a**b

        a = torch.from_numpy(a.data)
        b = b

        out_torch = a**b
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_matmul(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = a @ b

        a = torch.from_numpy(a.data)
        b = torch.from_numpy(b.data)

        out_torch = a @ b
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_transpose(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a @ b.T

        a = torch.from_numpy(a.data)
        b = torch.from_numpy(b.data)

        out_torch = a @ b.T
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_stack(self):
        a = [Tensor(np.random.rand(*SHAPE)) for _ in range(10)]

        out = Tensor.stack(a, 1)

        a = [torch.from_numpy(aa.data) for aa in a]

        out_torch = torch.stack(a, 1)

        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_conv2d(self):
        x = Tensor(np.random.rand(32, 6, 128, 128))
        f = Tensor(np.random.rand(8, 6, 3, 3))

        out = x.convolve2d(f)

        x = torch.from_numpy(x.data)
        f = torch.from_numpy(f.data)

        out_torch = F.conv2d(x, f)
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_reduce_sum(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.sum()

        a = torch.from_numpy(a.data)

        out_torch = a.sum()
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_sum_axis(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.sum(-1)

        a = torch.from_numpy(a.data)

        out_torch = a.sum(-1)
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_reshape(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.reshape(tuple(reversed(SHAPE)))

        a = torch.from_numpy(a.data)

        out_torch = a.reshape(tuple(reversed(SHAPE)))
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_tanh(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.tanh()

        a = torch.from_numpy(a.data)

        out_torch = a.tanh()
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_relu(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.relu()

        a = torch.from_numpy(a.data)

        out_torch = a.relu()
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_exp(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.exp()

        a = torch.from_numpy(a.data)

        out_torch = a.exp()
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_log(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.log()

        a = torch.from_numpy(a.data)

        out_torch = a.log()
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_sigmoid(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.sigmoid()

        a = torch.from_numpy(a.data)

        out_torch = a.sigmoid()
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

    def test_softmax(self):
        a = Tensor(np.random.rand(*SHAPE))

        out = a.softmax(-1)

        a = torch.from_numpy(a.data)

        out_torch = a.softmax(-1)
        self.assertTrue(np.allclose(out.data, out_torch.numpy()))

class TestGrad(unittest.TestCase):

    def test_add(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a + b
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = a_torch + b_torch
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_sub(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a - b
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = a_torch - b_torch
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_mul(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a * b
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = a_torch * b_torch
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_div(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a / b
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = a_torch / b_torch
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_pow(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = np.random.rand()

        out = a**b
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = b

        out_torch = a_torch**b_torch
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()))


    def test_matmul(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = a @ b
        out.backward()


        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = a_torch @ b_torch
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_transpose(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*SHAPE))

        out = a @ b.T
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = a_torch @ b_torch.T
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_stack(self):
        a = [Tensor(np.random.rand(*SHAPE)) for _ in range(10)]
        b = [Tensor(np.random.rand(*reversed(SHAPE))) for _ in range(10)]

        out = Tensor.stack([aa @ bb for aa, bb in zip(a, b)] , 1)
        out.backward()

        a_torch = [torch.tensor(aa.data, requires_grad=True) for aa in a]
        b_torch = [torch.tensor(bb.data, requires_grad=True) for bb in b]
        out_torch = torch.stack([aa @ bb for aa, bb in zip(a_torch, b_torch)], 1)
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a[0].grad, a_torch[0].grad.numpy()))

    def test_conv2d(self):
        x = Tensor(np.random.rand(32, 6, 128, 128))
        f = Tensor(np.random.rand(8, 6, 3, 3))

        out = x.convolve2d(f)
        out.backward()

        x_torch = torch.from_numpy(x.data)
        x_torch.requires_grad = True
        f_torch = torch.from_numpy(f.data)
        f_torch.requires_grad = True

        out_torch = F.conv2d(x_torch, f_torch)
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(x.grad, x_torch.grad.numpy()) and np.allclose(f.grad, f_torch.grad.numpy()))

    def test_reduce_sum(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).sum()
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).sum()
        out_torch.backward()

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_sum_axis(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).sum(-1)
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).sum(-1)
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_reshape(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).reshape((SHAPE[0]//2, 2*SHAPE[0]))
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).reshape((SHAPE[0]//2, 2*SHAPE[0]))
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_tanh(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).tanh()
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).tanh()
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_relu(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).relu()
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).relu()
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_exp(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).exp()
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).exp()
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_log(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).log()
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).log()
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_sigmoid(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).sigmoid()
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).sigmoid()
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))

    def test_softmax(self):
        a = Tensor(np.random.rand(*SHAPE))
        b = Tensor(np.random.rand(*reversed(SHAPE)))

        out = (a @ b).softmax(-1)
        out.backward()

        a_torch = torch.from_numpy(a.data)
        a_torch.requires_grad = True
        b_torch = torch.from_numpy(b.data)
        b_torch.requires_grad = True

        out_torch = (a_torch @ b_torch).softmax(-1)
        out_torch.backward(gradient=torch.ones_like(out_torch))

        self.assertTrue(np.allclose(a.grad, a_torch.grad.numpy()) and np.allclose(b.grad, b_torch.grad.numpy()))


if __name__ == '__main__':
    unittest.main()