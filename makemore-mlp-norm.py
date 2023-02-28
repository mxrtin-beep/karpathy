
# https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
# 30 features
# 17,000
# Multi-layer perceptron

# Input index for word (n/17,000)
# Table look-up in C
#	C is a table of 17,000 by 30 --> get that word's row of 30
#	C shared across all words
# hidden layer with tanh
# output layer with 17,000 neurons
# softmax layer (exponentiate, normalize)
# have the label --> maximize probability of outputting label

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', 'r').read().splitlines()

g = torch.Generator().manual_seed(2147483647)

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



# Building the dataset
# block_size is the context length: how many characters to take to predict the next

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


def normalize_tensor(ten, mean, std):

	return (ten - mean) / std

def get_split_data(words):

	random.seed(42)
	random.shuffle(words)
	n1 = int(0.8*len(words))
	n2 = int(0.9*len(words))

	X_tr, Y_tr = build_dataset(words[:n1], 3, N_WORDS)
	X_dev, Y_dev = build_dataset(words[n1:n2], 3, N_WORDS)
	X_te, Y_te = build_dataset(words[n2:], 3, N_WORDS)

	return X_tr, Y_tr, X_dev, Y_dev, X_te, Y_te


def get_parameters(g, block_size, n_inputs, n_features, hidden_layer_nodes):
	gain = (5/3)


	C = torch.randn((n_inputs, n_features),							 	generator=g) 

	# Squash W1 and b1 to void killing neurons and saturated tanh.
	W1 = torch.randn((n_features*block_size, hidden_layer_nodes), 		generator=g) * gain / ((n_features*block_size))**0.5
	b1 = torch.randn(hidden_layer_nodes, 								generator=g) * 0.01

	# Normally have wildly wrong weights --> squash them down to be closer to all  the same.
	# Can fix having extremely high loss on first iteration.
	W2 = torch.randn((hidden_layer_nodes, n_inputs), 					generator=g) * 0.01
	b2 = torch.randn(n_inputs, 											generator=g) * 0

	bngain = torch.ones((1, hidden_layer_nodes))
	bnbias = torch.zeros((1, hidden_layer_nodes))


	return [C, W1, W2, b2, bngain, bnbias]

def forward_normalize(X, parameters, mean, std):
	C, W1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	emb = C[X]
	embcat = emb.view(emb.shape[0], -1)

	# Biases are useless when doing batch norm on the same layer.
	# They will get extracted out by mean and bnbias.
	hpreact = embcat @ W1 # + b1

	hpreact = bngain * (hpreact - mean) / std + bnbias
	h = torch.tanh(hpreact)

	logits = h @ W2 + b2

	return logits

def forward_bn(X, parameters, bnmean_running, bnstd_running):
	C, W1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	emb = C[X] # (32, 3, 2); embed the characters into vectors
	embcat = emb.view(emb.shape[0], -1)	# concatenate the vectors
	hpreact = embcat @ W1 # + b1				# hidden layer preactivation

	bnmeani = hpreact.mean(0, keepdim=True)
	bnstdi = hpreact.std(0, keepdim=True)
	hpreact = normalize_tensor(hpreact, bnmeani, bnstdi)			# batch normalize
	hpreact = hpreact * bngain + bnbias		# scale and shift

	with torch.no_grad():
		bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
		bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi


	h = torch.tanh(hpreact) # (32, 100); hidden layer
	logits = h @ W2 + b2 # (32, 27); output layer

	return logits, bnmean_running, bnstd_running

def get_loss(logits, Y):
	return F.cross_entropy(logits, Y)


def backward(loss, parameters, learning_rate = 0.01):

	for p in parameters:
		p.grad = None
	loss.backward()

	for p in parameters:
		p.data += -1 * learning_rate * p.grad

	return parameters


def evaluate(X, Y, parameters, bnmean_running, bnstd_running, n_epochs, batch_size, learning_rate=0.1, dynamic_lr = False):

	for p in parameters:
		p.requires_grad = True

	
	C, W1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]
	


	for i in range(n_epochs):

		if dynamic_lr and i > (n_epochs / 10):
			learning_rate = 0.01

		ix = torch.randint(0, X.shape[0], (32,))

		logits, bnmean_running, bnstd_running = forward_bn(X[ix], parameters, bnmean_running, bnstd_running)
		loss = get_loss(logits, Y[ix])
		#print(loss.item())
	
		parameters = backward(loss, parameters, learning_rate)

		if i % (n_epochs / 10) == 0:
			print(f'{i:7d}/{n_epochs:7d}: {loss.item():.4f}')

		
	return parameters, bnmean_running, bnstd_running


# X should be Xtr
# Should only be called if normalize is true.
def get_whole_means(X, parameters):

	C, W1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	with torch.no_grad():
		# Pass training set through
		emb = C[X]
		embcat = emb.view(emb.shape[0], -1)
		hpreact = embcat @ W1 # + b1

		# measure mean and stdev over whole training set
		bnmean = hpreact.mean(0, keepdim=True)
		bnstd = hpreact.std(0, keepdim=True)

		return bnmean, bnstd

# Can easily overfit to 32 examples.
# Can't get 0 loss because cannot predict the first letter
# 100% correctly.

# Too slow to forward and backward for thousands of words each time.
# Use Batching.
# 	The quality of the gradient goes down, but it's much faster.
# 	Better to have a worse gradient and iterate 10x more.

# Can't sample with batch norm, because the stdev of 1 item is 0.
def sample(n_samples, block_size, parameters, seeded=False, mean=0, std=0):

	for i in range(n_samples):
		out = []
		context = [0]*block_size
		itos = get_itos(words)

		while True:

			X = torch.tensor([context])

			logits = forward_normalize(torch.tensor([context]), parameters, mean, std)

			probs = F.softmax(logits, dim=1)
			#print(probs)
			if seeded:
				ix = torch.multinomial(probs, num_samples=1, generator=g).item()
			else:
				ix = torch.multinomial(probs, num_samples=1).item()
			context = context[1:] + [ix]
			out.append(ix)

			if ix==0:
				break
		print(''.join(itos[i] for i in out))


# Another problem
# When you calculate y = x @ w, where x and w are Gaussian distributions,
# the stdev of y increases, which we don't want.
# Solution: divide y by n^0.5, where n is the number of input elements.
# Ex. x (1000, 10) @ w (10, 200) = y / 10^0.5.
# Use torch.nn.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
#	fan_in refers to whether or not to normalize activations or gradients (doesn't really matteR)
#	std = gain / sqrt(fan_mode)
#	nonlinearity affects gain (tanh: 5/3; linear or sigmoid: 1; relu: sqrt(2))
# 	gain is important for squashing functions to offset squashing and bring
#	stdev UP to 1.

# Makemore part 3
# Solving Problems
# 	Loss extremely high upon initialization --> scale down W2 and b2
# 	Saturated tanh / dead neurons --> scale down W1 and b1
# 	Not normalized --> kaiming normalization of weights by 5/3 / sqrt(fan_in)
# 	Batch Normalization --> normalize hidden layer activations to be Gaussian + scale and shift


def main():

	words = open('names.txt', 'r').read().splitlines()

	#g = torch.Generator().manual_seed(2147483647)
	seed = 21474836471
	#seed = torch.seed()
	print(seed)
	g = torch.Generator().manual_seed(seed)

	N_WORDS = len(words)

	parameters = get_parameters(g, block_size=3, n_inputs=27, n_features=10, hidden_layer_nodes=100)

	X_tr, Y_tr, X_dev, Y_dev, X_te, Y_te = get_split_data(words)

	bnmean_running = torch.zeros((1, 100))
	bnstd_running = torch.ones((1, 100))

	parameters, bnmean_running, bnstd_running = evaluate(X_tr, Y_tr, parameters, bnmean_running, bnstd_running, n_epochs=10000, batch_size=50, learning_rate=0.1, dynamic_lr=False)

	#bnmean, bnstd = get_whole_means(X_tr, parameters)
	
	logits = forward_normalize(X_dev, parameters, bnmean_running, bnstd_running)
	loss = get_loss(logits, Y_dev)
	

	print("Dev Loss: ", loss.item())




	sample(10, 3, parameters, seeded=False, mean=bnmean_running, std=bnstd_running)

	
main()

'''
clown = torch.tensor([[-1.263600,  0.32630, -0.91806]])
print(clown)
std = clown.std(0, keepdim=True)
print(std)
print(std[0, 0].item() != std[0, 0].item())
'''


