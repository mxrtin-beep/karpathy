
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np

# --------------------------------------- SUMMARY ---------------------------------------
'''
Let's say 4 examples, block_size = 8:

Xb			(4, 8)			4 words, 8 letters each
C[Xb]		(4, 8, 10)		Embeds each letter into 10 dimensions
Flatten		(4, 80)			Flattens char and dim layers
Linear		(4, 200)		Puts 80 dimensions into 200 hidden neurons (4, 80) @ (80, 200) + (200)

In matrix mult, torch treats first dimensions as batch dimensions, doesn't change
(4, 5, 6, 7, 80) @ (80, 200) = (4, 5, 6, 7, 200)

Problem with makemore is that it squishes eight characters down to one layer.
We want to group them in bigrams (20 char-dims instead of 80).
Don't want 80 flattened char-dims to come in, but separate (4, 4, 20) (4 groups of 2 characters * 10 features)
Want (4, 4, 20) @ (20, 200) + (200) = (4, 4, 200)

Changes:
	Linear layer shouldn't expect 80 inputs, but 20
	Flatten layer shouldn't flatten all the way to 80, but to 20


Flatten
	Input: e = (4, 8, 10)
	Performs: e.view(4, 80)
	Want: torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim = 2)


Embedding : (4, 8, 10)
FlattenConsecutive : (4, 4, 20)
Linear : (4, 4, 200)
BatchNorm1d : (4, 4, 200)
Tanh : (4, 4, 200)
FlattenConsecutive : (4, 2, 400)
Linear : (4, 2, 200)
BatchNorm1d : (4, 2, 200)
Tanh : (4, 2, 200)
FlattenConsecutive : (4, 400)
Linear : (4, 200)
BatchNorm1d : (4, 200)
Tanh : (4, 200)
Linear : (4, 27)


'''
block_size = 8


n_embd = 10
n_hidden = 100

max_steps = 50000
batch_size = 32

n_samples = 30

# --------------------------------------- BUILDING DATASET ---------------------------------------

g = torch.Generator().manual_seed(2147483647)
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0


itos = {i:s for s,i in stoi.items()}

vocab_size = len(itos) 	# 27

#random.seed(42)
random.shuffle(words)

def build_dataset(words):
	X, Y = [], []

	for w in words:

		context = [0] * block_size

		for ch in w + '.':
			ix = stoi[ch]
			X.append(context) # input: three chars
			Y.append(ix) # output: index of next char
			#print(''.join(itos[i] for i in context), '--->', itos[ix])
			context = context[1:] + [ix] # crop and append

	X = torch.tensor(X)
	Y = torch.tensor(Y)

	return X, Y


n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

X_tr, Y_tr = build_dataset(words[:n1])
X_dev, Y_dev = build_dataset(words[n1:n2])
X_te, Y_te = build_dataset(words[n2:])



# --------------------------------------- NETWORK CLASSES ---------------------------------------


class Linear:

	def __init__(self, fan_in, fan_out, bias=True):

		# Squash W1 to void killing neurons and saturated tanh.
		self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
		self.bias = torch.zeros(fan_out) if bias else None


	def __call__(self, x):
		self.out = x @ self.weight
		if self.bias is not None:
			self.out += self.bias

		return self.out

	def parameters(self):
		return [self.weight] + ([] if self.bias is None else [self.bias])


# Problem: only works for 1 dimension
# (32, 4, 68) --> (1, 4, 68); want to normalize 32*4 numbers
class BatchNorm1d:

	def __init__(self, dim, eps=1e-5, momentum=0.1):
		self.eps = eps
		self.momentum = momentum
		self.training = True

		# Parameters trained with backprop
		self.gamma = torch.ones(dim)	# Batch Norm Gain	
		self.beta = torch.zeros(dim)	# Batch Norm Bias

		# Buffers
		self.running_mean = torch.zeros(dim)
		self.running_var = torch.ones(dim)

	def __call__(self, x):

		# Forward Pass
		if self.training:


			if x.ndim == 2: # (32, 100) --> (1, 100)
				dim = 0
			elif x.ndim == 3: # (32, 4, 68) --> (1, 1, 68) instead of (1, 4, 68)
				dim = (0, 1)
				# Departure from pytorch: torch would batchnorm over (0th and 2nd layers, not 0th and 1st)

			xmean = x.mean(dim, keepdim=True) 				# Batch mean
			xvar = x.var(dim, keepdim=True, unbiased=True) 	# Batch variance
		else:
			xmean = self.running_mean
			xvar = self.running_var

		xhat = (x - xmean) / torch.sqrt(xvar + self.eps)	# Normalize to unit variance, avoid div by 0 with eps.
		self.out = self.gamma * xhat + self.beta

		# Update buffers (running mean and var).
		if self.training:
			with torch.no_grad():
				self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
				self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

		return self.out


	def parameters(self):
		return [self.gamma, self.beta] # Gain and bias


class Tanh:

	def __call__(self, x):
		self.out = torch.tanh(x)
		return self.out

	def parameters(self):
		return []

# C
class Embedding:

	def __init__(self, num_embeddings, embedding_dim):
		self.weight = torch.randn((num_embeddings, embedding_dim))

	def __call__(self, IX):
		self.out = self.weight[IX]
		return self.out

	def parameters(self):
		return [self.weight]


class FlattenConsecutive:

	def __init__(self, n):
		self.n = n 		# Number of elements to flatten and concatenate in last dimension

	def __call__(self, x):

		B, T, C = x.shape		# (4, 8, 10)
		# self.out = x.view(x.shape[0], -1) 	old

		# Input: (4, 8, 10); Output: (4, 4, 20)
		# First tensor: takes all even dimensions from first dimension (4, 4, 10)
		# Second tensor: takes all odd dimensions from first dimension (4, 4, 10)
		# Concat along second dim (10s): (4, 4, 20)

		#self.out = torch.cat([x[:, ::2, :], x[:, 1::2, :]], dim = 2)	# Explicit
		x = x.view(B, T//self.n, C*self.n)

		# Can happen if n is something like 3: superious dimension
		if x.shape[1] == 1:
			x = x.squeeze(1) 	# Will return (B, C*n)

		self.out = x
		return self.out

	def parameters(self):
		return []


class Sequential:

	def __init__(self, layers):
		self.layers = layers

	def __call__(self, x):

		for layer in self.layers:
			x = layer(x)
		self.out = x
		return self.out

	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters()]

	def print_layers(self):
		for layer in self.layers:
			print(layer.__class__.__name__, ':', tuple(layer.out.shape))


# --------------------------------------- NETWORK ARCHITECTURE ---------------------------------------

#torch.manual_seed(42)


# If you don't have the Tanh layers, your activations will explode.
# Also, your whole network will be one linear function.
model = Sequential([
	Embedding(vocab_size, n_embd), 
	FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
	FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
	FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
	Linear(n_hidden, vocab_size)
])


with torch.no_grad():
	# Last layer: make it less confident
	if isinstance(model.layers[-1], Linear):
		model.layers[-1].weight *= 0.1


parameters = model.parameters()
print("Number of parameters: ", sum(p.nelement() for p in parameters))
for p in parameters:
	p.requires_grad = True


# --------------------------------------- OPTIMIZATION ---------------------------------------


lossi = []


for i in range(max_steps):

	# Minibatch construction
	ix = torch.randint(0, X_tr.shape[0], (batch_size, ))
	Xb, Yb = X_tr[ix], Y_tr[ix]		# Batch X and Y

	# Forward pass
	logits = model(Xb)
	loss = F.cross_entropy(logits, Yb)		# loss function


	# Backward pass
	for p in parameters:
		p.grad = None
	loss.backward()


	# Update
	lr = 0.1 if i < (0.9 * max_steps) else 0.01 # Step learning rate decay
	for p in parameters:
		p.data += -lr * p.grad


	# Track stats
	if i % (max_steps * 0.1) == 0:
		print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
	lossi.append(loss.log10().item())


#model.print_layers()

# Split lossi into 200 rows of 1000 samples, take the mean of each row
lossi_avg = torch.tensor(lossi).view(-1, 1000).mean(1)	
# plt.plot(lossi_avg)
# plt.savefig('lossi_avg.png')
# --------------------------------------- VALIDATION ---------------------------------------

for layer in model.layers:
	layer.training = False

@torch.no_grad()
def split_loss(split):
	x, y = {
	'train': (X_tr, Y_tr),
	'val': (X_dev, Y_dev),
	'test': (X_te, Y_te),
	}[split]

	logits = model(x)
	loss = F.cross_entropy(logits, y)
	print(split, loss.item())

split_loss('train')
split_loss('val')
split_loss('test')


# --------------------------------------- SAMPLING ---------------------------------------


for _ in range(n_samples):

	out = []
	context = [0] * block_size
	while True:

		# Forward
		x = torch.tensor([context])
		logits = model(x)
		probs = F.softmax(logits, dim=1)

		# Sample
		ix = torch.multinomial(probs, num_samples=1).item()

		# shift context window and track samples
		context = context[1:] + [ix]
		out.append(ix)

		# break if we sample the end token
		if ix == 0:
			break

	print(''.join(itos[i] for i in out))


torch.save(model.parameters(), 'parameters.txt')


p = torch.load('parameters.txt')
