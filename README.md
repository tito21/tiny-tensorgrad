# Tiny-tensorgrad

A tiny deep learning framework using numpy as backend. It has a tensor autograd
engine inspired by Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) and a pytorch like API

![Results of a simple classification problem](demo.png)

## Why another deep learning framework?

There's no need to for a new deep learning framework but it is useful as a
learning tool.

## Demo

The file `demo.ipynb` has a demo classification problem. And in
`demo-mnist.ipynb` a demo of classifying MNIST digits.

## Test

Run test against pytorch

`python test.py`

## TODO

- [x] Add sigmoid, softmax and train with cross entropy
- [x] Add convolutional layers
- [x] Test on MNIST
- [x] Add optimizers