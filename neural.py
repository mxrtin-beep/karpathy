import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import random


class Value:

	def __init__(self, data, _children={}, _op='', label=''):
		self.data = data
		self._prev = set(_children)
		self._backward = lambda: None
		self._op = _op
		self.grad = 0.0
		self.label = label

	def __repr__(self):
		return f"Value({self.data})"


	# When you perform an operation, you store the previous values as children.
	# Also store the operation.
	def __add__(self, other):

		# Changes to primitive if trying to add a primitive
		other = other if isinstance(other, Value) else Value(other)

		out = Value(self.data + other.data, (self, other), "+")

		def _backward():
			self.grad += 1.0 * out.grad
			other.grad += 1.0 * out.grad


		out._backward = _backward
		return out

	def __radd__(self, other):
		return self * other

	def __neg__(self):
		return self * -1

	def __sub__(self, other):
		return self + (-other)

	def __mul__(self, other):

		other = other if isinstance(other, Value) else Value(other)

		out = Value(self.data * other.data, (self, other), "*")

		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad

		out._backward = _backward

		return out

	# Reverses self and other
	def __rmul__(self, other):
		return self * other


	def __truediv__(self, other):
		return self * other**-1


	def __pow__(self, other):
		assert isinstance(other, (int, float)) # other must be int or float
		out = Value(self.data**other, (self,), f'**{other}')

		def _backward():
			self.grad += other * self.data ** (other-1) * out.grad

		out._backward = _backward

		return out

	def tanh(self):

		n = self.data
		t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
		out = Value(t, (self, ), 'tanh')

		def _backward():
			self.grad += (1 - t**2) * out.grad

		out._backward = _backward
		return out

	def exp(self):
		x = self.data
		out = Value(math.exp(x), (self, ), 'exp')

		def _backward():
			self.grad += self.data * out.grad

		out._backward = _backward

		return out


	def backward(self):

		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self)

		self.grad = 1.0
		for node in reversed(topo):
			node._backward()


def trace(root):
	nodes, edges = set(), set()

	def build(v):
		if v not in nodes:
			nodes.add(v)
			for child in v._prev:
				edges.add((child, v))
				build(child)
	build(root)
	return nodes, edges

def draw_dot(root):
	dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

	nodes, edges = trace(root)
	for n in nodes:
		uid = str(id(n))

		dot.node(name = uid, label = "{%s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

		if n._op:
			dot.node(name = uid + n._op, label= n._op)
			dot.edge(uid + n._op, uid)

	for n1, n2 in edges:
		dot.edge(str(id(n1)), str(id(n2)) + n2._op)

	dot.render('doctest-output/round-table.gv', view=True)
	return dot



class Neuron:


	def __init__(self, nin):
		self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
		self.b = Value(random.uniform(-1, 1))

	# x is prev layer
	def __call__(self, x):
		# w * x + b

		# sum function takes in start value as second parameter (bias)
		act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
		out = act.tanh()

		return out

	def parameters(self):
		return self.w + [self.b]

class Layer:

	def __init__(self, nin, nout):
		self.neurons = [Neuron(nin) for _ in range(nout)]

	def __call__(self, x):
		outs = [n(x) for n in self.neurons]
		return outs[0] if len(outs) == 1 else outs

	def parameters(self):
		params = []
		for neuron in self.neurons:
			ps = neuron.parameters()
			params.extend(ps)
		return params


class MLP():

	def __init__(self, nin, nouts):
		sz = [nin] + nouts
		self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

	def parameters(self):

		params = []
		for layer in self.layers:
			ps = layer.parameters()
			params.extend(ps)
		return params



x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
print(n(x))
#draw_dot(n(x))

xs = [
	[2.0, 3.0, -1.0],
	[3.0, -1.0, 0.5],
	[0.5, 1.0, 1.0],
	[1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(20):
	# forward
	ypred = [n(x) for x in xs]
	loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

	# backward
	for p in n.parameters():
		p.grad = 0.0 # can't let grads accumulate over multiple passes 
	loss.backward()

	# update
	for p in n.parameters():
		p.data += -0.05 * p.grad

	print(k, loss.data)

print(ypred)

