
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# --------------------------------------- BUILDING DATASET ---------------------------------------

g = torch.Generator().manual_seed(2147483647)
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0


itos = {i:s for s,i in stoi.items()}

vocab_size = len(itos) 	# 27
block_size = 3

random.seed(42)
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
			xmean = x.mean(0, keepdim=True) 				# Batch mean
			xvar = x.var(0, keepdim=True, unbiased=True) 	# Batch variance
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


class Flatten:

	def __call__(self, x):
		self.out = x.view(x.shape[0], -1)
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


# --------------------------------------- NETWORK ARCHITECTURE ---------------------------------------

torch.manual_seed(42)

n_embd = 10
n_hidden = 200

# If you don't have the Tanh layers, your activations will explode.
# Also, your whole network will be one linear function.
model = Sequential([
	Embedding(vocab_size, n_embd), Flatten(),
	Linear((n_embd * block_size), n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
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


max_steps = 20000
batch_size = 32
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


# --------------------------------------- SAMPLING ---------------------------------------


for _ in range(10):

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
