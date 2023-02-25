
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

X, Y = build_dataset(words, 3, N_WORDS)
# X: 32x3
#	32 examples
#	Each input is 3 integers
# Y: 32x1
#	32 examples
#	Each output is 1 integer (label)


# Lookup table

C = torch.randn((27, 2), generator=g) 	# embedding into 2 dimensions

#val = F.one_hot(torch.tensor(5), num_classes=27).float() @ C
# 1x27 @ 27x2 = 1x2
# Equivalent to lookup table as well as layer in a NN, where C is the weights matrix.

# Can index tensors with tensors
# Returns a 32x3x2 tensor
# The 32x3 is the X matrix, and for each value there are 2 features from C. 
emb = C[X]

# For example, X[13, 2] == 1. C[X][13, 2] == (a, b). C[1] == (a, b).

# --------------------------------- Hidden Layer ---------------------------------

# 6 features (2 features per char x 3 chars) to 100 nodes.
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)

# Cannot do emb @ W1 + b1, because different indices.

#word1 = emb[:, 0, :] 	# Plucks out all examples, 0th index, all features.
						# 32x2 matrix.


# These all do the same thing:
#clown = torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], dim = 1)
#clown2 = torch.cat(torch.unbind(emb, 1), 1)
# Concatenates all couples of features of each letter together.
# 32x6

# Can also do tensor.view(a, b, c)
# As long as a, b, and c multiply to length of tensor, it works.
# VERY efficient, doesn't change storage or memory of tensor.

# -1 infers the size to be 32 so it multiplies to 192.
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (-1, 1)

# 32x6 @ 6x100 = 32x100
# 100-dimensional activation for 32 examples.

# --------------------------------- Output Layer ---------------------------------

# 100 nodes to 27 logits
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

logits = h @ W2 + b2
# Shape is 32x27

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
# Every row sums to 1.

# --------------------------------- Labels ---------------------------------


# For each row of prob (0 to 31), index the Y column.
# Probability of the next letter occuring.
# Ideally should all be 1.
#prob[torch.arange(32), Y] # Probability of each char being next.

#loss = -prob[torch.arange(32), Y].log().mean()

# Can just use tensorflow function to do the last three lines.
# Much more time and space efficient.
# Also avoids logits.exp() going to infinity for big numbers.
#	Since you can add any number to all the logits and it will preserve the answer,
# 	Tensorflow calculates the max and subtracts it.
loss = F.cross_entropy(logits, Y)

parameters = [C, W1, b1, W2, b2]


def get_split_data(words):

	random.seed(42)
	random.shuffle(words)
	n1 = int(0.8*len(words))
	n2 = int(0.9*len(words))

	X_tr, Y_tr = build_dataset(words[:n1], 3, N_WORDS)
	X_dev, Y_dev = build_dataset(words[n1:n2], 3, N_WORDS)
	X_te, Y_te = build_dataset(words[n2:], 3, N_WORDS)

	return X_tr, Y_tr, X_dev, Y_dev, X_te, Y_te

X_tr, Y_tr, X_dev, Y_dev, X_te, Y_te = get_split_data(words)

def get_parameters(g, block_size, n_inputs, n_features, hidden_layer_nodes):
	C = torch.randn((n_inputs, n_features), generator=g) 

	W1 = torch.randn((n_features*block_size, hidden_layer_nodes), generator=g)
	b1 = torch.randn(hidden_layer_nodes, generator=g)

	W2 = torch.randn((hidden_layer_nodes, n_inputs), generator=g)
	b2 = torch.randn(n_inputs, generator=g)

	return [C, W1, b1, W2, b2]

parameters = get_parameters(g, block_size=3, n_inputs=27, n_features=10, hidden_layer_nodes=100)

def forward(X, Y, parameters):

	C, W1, b1, W2, b2 = [parameters[i] for i in range(len(parameters))]

	emb = C[X] # (32, 3, 2)
	h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
	logits = h @ W2 + b2 # (32, 27)
	loss = F.cross_entropy(logits, Y)

	return loss

def backward(loss, parameters, learning_rate = 0.01):

	for p in parameters:
		p.grad = None
	loss.backward()

	for p in parameters:
		p.data += -1 * learning_rate * p.grad

	return parameters


def evaluate(X, Y, parameters, n_epochs, batch_size, learning_rate=0.1, dynamic_lr = False):

	for p in parameters:
		p.requires_grad = True

	C, W1, b1, W2, b2 = [parameters[i] for i in range(len(parameters))]

	for i in range(n_epochs):

		if i > (n_epochs / 2):
			learning_rate = 0.01

		ix = torch.randint(0, X.shape[0], (32,))

		loss = forward(X[ix], Y[ix], parameters)
		#print(loss.item())
	
		parameters = backward(loss, parameters, learning_rate)

	return parameters


# Can easily overfit to 32 examples.
# Can't get 0 loss because cannot predict the first letter
# 100% correctly.
parameters = evaluate(X_tr, Y_tr, parameters, n_epochs=5000, batch_size=32, learning_rate=0.1, dynamic_lr=False)

loss = forward(X_dev, Y_dev, parameters)

print("Dev Loss: ", loss.item())
# Too slow to forward and backward for thousands of words each time.
# Use Batching.
# 	The quality of the gradient goes down, but it's much faster.
# 	Better to have a worse gradient and iterate 10x more.

def sample(n_samples, block_size, parameters):

	for i in range(n_samples):
		out = []
		context = [0]*block_size
		itos = get_itos(words)
		C, W1, b1, W2, b2 = [parameters[i] for i in range(len(parameters))]

		while True:

			emb = C[torch.tensor([context])]
			h = torch.tanh(emb.view(1, -1) @ W1 + b1)
			logits = h @ W2 + b2 
			probs = F.softmax(logits, dim=1)
			ix = torch.multinomial(probs, num_samples=1, generator=g).item()
			context = context[1:] + [ix]
			out.append(ix)

			if ix==0:
				break

		print(''.join(itos[i] for i in out))

sample(10, 3, parameters)


