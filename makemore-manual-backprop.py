
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
	b1 = torch.randn(hidden_layer_nodes, 								generator=g) * 0.1

	# Normally have wildly wrong weights --> squash them down to be closer to all  the same.
	# Can fix having extremely high loss on first iteration.
	W2 = torch.randn((hidden_layer_nodes, n_inputs), 					generator=g) * 0.1
	b2 = torch.randn(n_inputs, 											generator=g) * 0.1

	bngain = torch.randn((1, hidden_layer_nodes)) * 0.1 + 1.0
	bnbias = torch.randn((1, hidden_layer_nodes)) * 0.1

	parameters = [C, W1, b1, W2, b2, bngain, bnbias]

	for p in parameters:
		p.requires_grad = True


	return parameters

def forward_normalize(X, parameters, mean, std):
	C, W1, b1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	emb = C[X]
	embcat = emb.view(emb.shape[0], -1)

	# Biases are useless when doing batch norm on the same layer.
	# They will get extracted out by mean and bnbias.
	hpreact = embcat @ W1 + b1

	hpreact = bngain * (hpreact - mean) / std + bnbias
	h = torch.tanh(hpreact)

	logits = h @ W2 + b2

	return logits

def get_loss(logits, Y):
	n = 32
	logit_maxes = logits.max(1, keepdim=True).values
	norm_logits = logits - logit_maxes 		# subtract max for numerical stability
	counts = norm_logits.exp()
	counts_sum = counts.sum(1, keepdim=True)
	counts_sum_inv = counts_sum**-1
	probs = counts * counts_sum_inv
	logprobs = probs.log()
	loss = -logprobs[range(n), Y].mean()

	params = [	
		logprobs, probs, counts, counts_sum, counts_sum_inv,
		norm_logits, logit_maxes
	]
	return loss, params


def forward_bn(X, parameters, Y):
	n = 32
	C, W1, b1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	emb = C[X] # (32, 3, 2); embed the characters into vectors
	embcat = emb.view(emb.shape[0], -1)	# concatenate the vectors


	# Linear Layer 1
	hprebn = embcat @ W1 + b1			

	# BatchNorm Layer
	bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
	bndiff = hprebn - bnmeani
	bndiff2 = bndiff**2

	bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
	bnvar_inv = (bnvar + 1e-5)**(-0.5)
	bnraw = bndiff * bnvar_inv
	hpreact = bngain * bnraw + bnbias

	# Non-linearity
	h = torch.tanh(hpreact)

	# Linear layer 2
	logits = h @ W2 + b2 	# Output layer

	# Cross-Entropy Loss
	loss, loss_params = get_loss(logits, Y)

	params = [
		logits, h, hpreact, bnraw, bnvar_inv, bnvar,
		bndiff2, bndiff, hprebn, bnmeani, embcat, emb
	]

	total_params = params + loss_params
	return loss, logits, total_params


def backward(loss, parameters, params, learning_rate = 0.01):

	for p in parameters:
		p.grad = None
	
	for t in params:
		t.retain_grad()

	loss.backward()

	#for p in parameters:
	#	p.data += -1 * learning_rate * p.grad

	return parameters, params


def backprop(params, parameters, Y):

	n = 32
	logits, h, hpreact, bnraw, bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani, embcat, emb, logprobs, probs, counts, counts_sum, counts_sum_inv, norm_logits, logit_maxes = [params[i] for i in range(len(params))]

	C, W1, b1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	# Y returns the column for each row.
	# Logprobs[range(n), Y] plucks out the logprobs of the next character in a sequence.
	# Y is 32x1 tensors, because 32 examples
	# Logprobs is 32x27, but only 32 of them participate (Logprobs[range(n), Y] is 32x1)

	# Loss = -logprobs[range(n), Y].mean()
	# Loss = -(a + b + c) / 3 = -1/3*a -1/3b - 1/3c
	# dLoss/da = -1/3
	# dLoss/da = -1/n
	dlogprobs = torch.zeros_like(logprobs) 	# same dimensions as logprobs
	dlogprobs[range(n), Y] = -1.0 / n
	cmp('logprobs', dlogprobs, logprobs)


	# Logprobs = probs.log()
	# dLoss/dprobs = dLoss/dLogprobs * dLogprobs/dLogs
	# y = ln(x)
	# dy/dx = 1/x
	dprobs = (1.0 / probs) * dlogprobs
	cmp('probs', dprobs, probs)


	# Probs = counts(32,27) * counts_sum_inv(32x1)
	# dLoss/dCSI = dLoss/dProbs * dProbs/dCSI
	# y = a * b; dy/db = a = counts
	d_counts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
	cmp('counts_sum_inv', d_counts_sum_inv, counts_sum_inv)

	# dLoss/dCounts = dLoss/dProbs * dProbs/dCounts
	# Probs = counts_sum_inv * counts
	# dProbs/dCounts = counts_sum_inv
	d_counts = counts_sum_inv * dprobs
	# Not done! counts is used in counts_sum and probs, so more contribution later.

	# dLoss/dCounts_sum = dLoss/dCSI * dCSI/dCounts_sum
	# CSI = 1 / countssum
	# y = 1 / x
	# dy/dx = -1/x^2
	d_counts_sum = (-1) * counts_sum**(-2) * d_counts_sum_inv
	cmp('counts_sum', d_counts_sum, counts_sum)

	# Counts(32x27); Counts_sum(32x1)
	# Counts_Sum sums up the rows into a column tensor
	# a1, a2, a3 --> b1 (=a1 + a2 + a3)
	# a4, a5, a6 --> b2 (=a4 + a5+ a6)
	# Have the derivative with respect to b1 (d_counts_sum), want
	# derivative with respect to as
	# db1/da1 = 1.
	# Want to take d_counts_sum(32x1) and replicate it 27 times.
	d_counts += torch.ones_like(counts) * d_counts_sum
	cmp('counts', d_counts, counts)

	# dLoss/dNL = dLoss/dCounts * dCounts/dNL
	# Counts = NL.exp()
	# y = e^x
	# dy/dx = e^x
	d_norm_logits = counts * d_counts
	cmp('norm_logits', d_norm_logits, norm_logits)

	# norm_logits(32, 27) = logits(32, 27) - logit_maxes(32, 1)
	# c = a - b
	# c11 c12 c13 = a11 a12 a13  -  b1
	# 

	# logits and norm_logits are same shape, so derivative
	# just gets passed through because it's addition
	# dc11/da11 = 1
	# Not done! Logits is used in max_logits.
	d_logits = d_norm_logits.clone()

	# dc11/db1 = -1, but keep reusing it for dc12, dc13, etc.
	# So you have to do a sum
	# Logit maxes is there to prevent overflow, but it doesn't actually
	# impact the loss, so d_logit_maxes is very small to 0.
	d_logit_maxes = (-d_norm_logits).sum(1, keepdim=True)
	cmp('logit_maxes', d_logit_maxes, logit_maxes)

	# Logit_maxes uses logits: finds the max of each row and plucks it out.
	# dlogits is 1 * the local derivative of the value plucked out * dlogit_maxes
	# we need to scatter it
	d_logits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * d_logit_maxes
	# The one_hot will make an array with 32 rows and 27 columns
	# all zeros except where logits.max(1) is (one per row).
	# num_classes = 27
	# Multiply 32x27 with 32x1 column of d_logit_maxes, which will get broadcast to 32x27 and
	# elementwise multiply
	cmp('logits', d_logits, logits)

	# logits(32,27) = h(32,64) @ W2(64,27) + b2(27)
	# d = a@b + c
	# d11 = a11b11 + b12*b21 + c1
	# d12 = a11b12 + a12*b22 + c2
	# We have dL/dd11 and dL/dd12 (d_logits)

	# dL/da11 = dL/dd11 * dd11/da11 + dL/dd12 * dd12/da11
	# 	dd11/da11 = b11
	# 	dd12/da11 = b12
	# dL/da11 = dL/dd11 * b11 + dL/dd12 * b12

	# Generalized
	# dL/da = dL/dd * b(transposed)
	# dL/db = a(transposed) * dL/dd

	# Also, shape of d_h has to be the same as h (32, 64)
	# Have to multiply it by d_logits(32, 27), so you have to
	# multiply d_logits by (27, 64) matrix.
	# The only (27,64) matrix we have is W2 transposed
	d_h = d_logits @ W2.T
	cmp('h', d_h, h)

	# d = a@b + c
	# logits = h @ W2 + b2
	# dL/dc1 = dL/dd11 * dd11/dc1 + dL/dd21 * dd21/dc1
	# dL/dc1 = dL/dd11 * 1 + dL/dd21 * 1
	# General: dL/dc = dL/dd.sum(0) (sum across columns)
	# dL/dc_i = dL/dd1i + dL/dd2i + ...

	# Also, dW2 must be (64, 27), must
	# come from a matrix multiplication with d_logits(32,27) and h(32,64)
	# HAS to be h.T @ d_logits
	d_W2 = h.T @ d_logits
	cmp('W2', d_W2, W2)

	# db2 has to be 27, sum of d_logits.
	# HAS to be direction of 0 because you need to eliminate the 32 dimension
	d_b2 = d_logits.sum(0)
	cmp('b2', d_b2, b2)

def cmp(s, dt, t):
	ex = torch.all(dt == t.grad).item()		# Exact result
	app = torch.allclose(dt, t.grad)			# Checks for approximately close result
	maxdiff = (dt - t.grad).abs().max().item()	# Max difference
	print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')




def evaluate(X, Y, parameters, bnmean_running, bnstd_running, n_epochs, batch_size, learning_rate=0.1, dynamic_lr = False):

	for p in parameters:
		p.requires_grad = True

	
	C, W1, b1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]
	


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

	C, W1, b1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	with torch.no_grad():
		# Pass training set through
		emb = C[X]
		embcat = emb.view(emb.shape[0], -1)
		hpreact = embcat @ W1 + b1

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

	batch_size = 32
	ix = torch.randint(0, X_tr.shape[0], (batch_size, ), generator=g)
	Xb, Yb = X_tr[ix], Y_tr[ix]

	loss, logits, params = forward_bn(Xb, parameters, Yb)
	
	parameters, params = backward(loss, parameters, params)
	backprop(params, parameters, Yb)

main()

'''
clown = torch.tensor([[-1.263600,  0.32630, -0.91806]])
print(clown)
std = clown.std(0, keepdim=True)
print(std)
print(std[0, 0].item() != std[0, 0].item())
'''


