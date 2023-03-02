
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


g = torch.Generator().manual_seed(2147483647)
words = open('names.txt', 'r').read().splitlines()
N_WORDS = len(words)

def get_stoi(words):
	chars = sorted(list(set(''.join(words))))
	stoi = {s:i+1 for i,s in enumerate(chars)}
	stoi['.'] = 0

	return stoi

def get_itos(words):
	stoi = get_stoi(words)
	itos = {i:s for s,i in stoi.items()}
	return itos

def build_dataset(words, block_size, n_words):

	X, Y = [], []
	stoi = get_stoi(words)
	itos = get_itos(words)

	for w in words[:n_words]:

		#print(w)
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

def get_split_data(words):

	random.seed(42)
	random.shuffle(words)
	n1 = int(0.8*len(words))
	n2 = int(0.9*len(words))

	X_tr, Y_tr = build_dataset(words[:n1], 3, N_WORDS)
	X_dev, Y_dev = build_dataset(words[n1:n2], 3, N_WORDS)
	X_te, Y_te = build_dataset(words[n2:], 3, N_WORDS)

	return X_tr, Y_tr, X_dev, Y_dev, X_te, Y_te

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

# n_embd: 		dimensionality of character embedding vectors; number of features per character
# n_hidden:		number of neurons in hidden layer
def get_parameters(vocab_size = 27, block_size = 3, n_embd = 10, n_hidden = 100):


	C = torch.randn((vocab_size, n_embd), generator=g)

	# If you don't have the Tanh layers, your activations will explode.
	# Also, your whole network will be one linear function.
	layers = [
		Linear((n_embd * block_size), n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
		Linear(				n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
		Linear(				n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
		Linear(				n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
		Linear(				n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
		Linear(				n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size), 
	]

	with torch.no_grad():
		# Last layer: make it less confident
		if isinstance(layers[-1], Linear):
			layers[-1].weight *= 0.1

		elif isinstance(layers[-1], BatchNorm1d):
			layers[-1].gamma *= 0.1

		# All other layers: apply gain
		for layer in layers[:-1]:
			if isinstance(layer, Linear):
				layer.weight *= 5/3

	parameters = [C] + [p for layer in layers for p in layer.parameters()]
	print("Number of parameters: ", sum(p.nelement() for p in parameters))
	for p in parameters:
		p.requires_grad = True


	return C, layers, parameters


def train(C, layers, parameters, n_steps = 1000, batch_size = 32):
	lossi = []
	ud = []		# update to data ratio

	X_tr, Y_tr, X_dev, Y_dev, X_te, Y_te = get_split_data(words)


	for i in range(n_steps):

		# Minibatch construction
		ix = torch.randint(0, X_tr.shape[0], (batch_size, ), generator=g)
		Xb, Yb = X_tr[ix], Y_tr[ix]		# Batch X and Y

		# Forward pass
		emb = C[Xb]							# embed characters into vectors
		x = emb.view(emb.shape[0], -1)		# concatenate vectors
		for layer in layers:				# pass x through layers
			x = layer(x)
		loss = F.cross_entropy(x, Yb)		# loss function


		# Backward pass
		for layer in layers:
			layer.out.retain_grad()
		for p in parameters:
			p.grad = None
		loss.backward()


		# Update
		lr = 0.1 if i < (0.9 * n_steps) else 0.01 # Step learning rate decay
		for p in parameters:
			p.data += -lr * p.grad


		# Track stats
		if i % (n_steps * 0.1) == 0:
			print(f'{i:7d}/{n_steps:7d}: {loss.item():.4f}')
		lossi.append(loss.log10().item())

		# Ratio of standard deviation of update to standard deviation of data.
		with torch.no_grad():
			ud.append([(lr*p.grad.std() / p.data.std()).log().item() for p in parameters])

	return C, layers, parameters, lossi, ud


def visualize_layers(layers):
	plt.figure()
	legends=[]

	for i, layer in enumerate(layers[:-1]): 	# exclude output layer
		if isinstance(layer, Tanh):
			t = layer.out
			print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
			hy, hx = torch.histogram(t, density=True)
			plt.plot(hx[:-1].detach(), hy.detach())
			legends.append(f'layer {i} ({layer.__class__.__name__}')
	plt.legend(legends)
	plt.title('Activation Distribution')
	plt.savefig('layers.png')

def visualize_gradients(layers):
	plt.figure()
	legends=[]

	for i, layer in enumerate(layers[:-1]): 	# exclude output layer
		if isinstance(layer, Tanh):
			t = layer.out.grad
			print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
			hy, hx = torch.histogram(t, density=True)
			plt.plot(hx[:-1].detach(), hy.detach())
			legends.append(f'layer {i} ({layer.__class__.__name__}')
	plt.legend(legends)
	plt.title('Activation Distribution')
	plt.savefig('gradients.png')

def visualize_weights(parameters):
	plt.figure()
	legends = []
	for i, p in enumerate(parameters):
		t = p.grad
		if p.ndim == 2:
			print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
			hy, hx = torch.histogram(t, density=True)
			plt.plot(hx[:-1].detach(), hy.detach())
			legends.append(f'{i} {tuple(p.shape)}')
	plt.legend(legends)
	plt.title("Weights Gradient Distribution")
	plt.savefig('weights.png')


def visualize_ud(parameters, ud):
	plt.figure()
	legends = []
	for i, p in enumerate(parameters):
		if p.ndim == 2:
			plt.plot([ud[j][i] for j in range(len(ud))])
			legends.append('param %d' % i)
	plt.plot([0, len(ud)], [-3, -3], 'k')	# Values should be roughly 1e-3
	plt.legend(legends)
	plt.title("Update to data Distribution")
	plt.savefig('ud.png')




def main():

	C, layers, parameters = get_parameters(vocab_size = 27, block_size = 3, n_embd = 10, n_hidden = 100)

	C, layers, parameters, lossi, ud = train(C, layers, parameters)

	visualize_layers(layers)
	visualize_gradients(layers)
	visualize_weights(parameters)
	visualize_ud(parameters, ud)


main()