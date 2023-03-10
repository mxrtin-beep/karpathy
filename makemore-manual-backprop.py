
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

	# return F.cross_entropy(logits, Y)		# Almost exact same loss, but much faster.
	return loss, params

def fast_loss(logits, Y):
	return F.cross_entropy(logits, Y)


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

	# Why divide by n-1 instead of n?
	# n-1 is unbiased, n is biased
	# n-1 is better for samples, which is the case for us because we use minibatches
	# the paper uses a mismatch
	# unfortunately, pytorch's batchnorm uses the biased version
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

def fast_forward(X, parameters, sample=False):
	n = 32
	C, W1, b1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	emb = C[X] # (32, 3, 2); embed the characters into vectors
	embcat = emb.view(emb.shape[0], -1)	# concatenate the vectors


	# Linear Layer 1
	hprebn = embcat @ W1 + b1

	#hpreact = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True))
	bnmean = hprebn.mean(0, keepdim=True)

	if sample:
		bnvar = hprebn.var(0, keepdim=True, unbiased=False)
		if bnvar[0,0] != 0:
			bnvar = hprebn.var(0, keepdim=True, unbiased=True)
	else:
		bnvar = hprebn.var(0, keepdim=True, unbiased=True)
	bnvar_inv = (bnvar + 1e-5)**-0.5
	bnraw = (hprebn - bnmean) * bnvar_inv
	hpreact = bngain * bnraw + bnbias

	h = torch.tanh(hpreact)

	# Linear layer 2
	logits = h @ W2 + b2 	# Output layer

	# Cross-Entropy Loss
	#loss = F.cross_entropy(logits, Y)

	fast_params = [logits, h, hpreact, bnraw, bnvar_inv, bnvar, hprebn, bnmean, embcat, emb]

	return fast_params, logits

def backward(loss, parameters, params, learning_rate = 0.01):

	for p in parameters:
		p.grad = None
	
	for t in params:
		t.retain_grad()

	loss.backward()

	# Commented this out because it fucked with backprop
	#for p in parameters:
	#	p.data += -1 * learning_rate * p.grad

	return parameters, params


def backprop(params, parameters, X, Y):

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

	#fast_d_logits = fast_loss_backprop(logits, Y)
	#cmp('fast logits', fast_d_logits, logits)

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


	# dLoss/dhpreact = dLoss/dh * dh/dhpreact
	# h = tanh(hpreact)
	# y = tanh(x)
	# dy/dx = 1 - tanh(x)^2
	# dh/dhpreact = 1 - h^2
	d_hpreact = (1 - h**2) * d_h
	cmp('hpreact', d_hpreact, hpreact)

	# dLoss/dbngain = dLoss/dhpreact * dhpreact/dbngain
	# hpreact(32,64) = bngain(1,64) * bnraw(32,64) + bnbias(1,64) (scale and shift)
	# bngain is casted as 32,64 to elementwise multiply bnraw
	# dhpreact/dbngain = bnraw
	d_bngain = (bnraw * d_hpreact).sum(0, keepdim=True)	# (32,64)*(32,64) should be a (1,64) --> sum across examples
	cmp('bngain', d_bngain, bngain)

	# dLoss/bnraw = dLoss/hpreact * dhpreact/dbnraw
	# dhpreact/dbnraw = bngain
	d_bnraw = bngain * d_hpreact		# (32,64) = (1,64)*(32,64)
	cmp('bnraw', d_bnraw, bnraw)

	# dLoss/dbnbias = dLoss/hpreact * dhpreact/dbnbias
	# dhpreact/dbnbias = 1
	d_bnbias = (1.0 * d_hpreact).sum(0, keepdim=True)	# (1, 64) = 1.0 * (32,64)
	cmp('bnbias', d_bnbias, bnbias)

	# dLoss/dbndiff = dLoss/d_bnraw * d_bnraw/d_bndiff
	# bnraw(32,64) = bndiff(32,64) * bnvar_inv(1,64)
	d_bndiff = bnvar_inv * d_bnraw
	d_bnvar_inv = (bndiff * d_bnraw).sum(0, keepdim=True) #bnvar-inv needs to be 1,64, but is 32,64
	#cmp('bndiff', d_bndiff, bndiff)	# bndiff is not finished because it is used earlier
	cmp('bnvar_inv', d_bnvar_inv, bnvar_inv)


	# dLoss/dbnvar = dLoss/dbnvar_inv * dbnvar_inv/dbnvar
	# bnvar_inv = (bnvar + 1e-5)**(-0.5)
	# dbnvar_inv/dbnvar = -0.5 * (bnvar + 1.e-5)**(-1.5)
	d_bnvar = d_bnvar_inv * ((-0.5) * (bnvar + 1e-5)**(-1.5))
	cmp('bnvar', d_bnvar, bnvar)


	# bnvar(1, 64), bndiff2(32, 64)
	# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
	# y = 1/(n-1) * x.sum(0) (sum over columns)
	# sum in the forward pass = replication in the backward pass
	# a11 a12
	# a21 a22
	# --->
	# b1, b2
	# b1 = 1/(n-1)*(a11 + a21) (32 times)
	# b2 = 1/(n-1)*(a21 + a22)
	# db1/da = 1/(n-1) * number of as
	d_bndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * d_bnvar  # (32, 64)*(1, 64)
	cmp('bndiff2', d_bndiff2, bndiff2)

	# dLoss/dbndiff = dLoss/dbndiff2 * dbndiff2/dbndiff
	# bndiff2 = bndiff^2
	# dbndiff2/dbndiff = 2 * bndiff
	d_bndiff += (d_bndiff2 * 2 * bndiff)
	cmp('bndiff', d_bndiff, bndiff)


	# dLoss/dbnmeani = dLoss/dbndiff * dbndiff/dbnmeani
	# bndiff = hprebn - bnmeani
	# dbndiff/dbnmeani = -1
	d_bnmeani = (-1.0) * (d_bndiff.sum(0, keepdim=True)) # (1, 64), (32,64)
	cmp('bnmeani', d_bnmeani, bnmeani)


	# dLoss/dhprebn = dLoss/dbndiff * dbndiff/dhprebn + dLoss/dbnmeani * dbnmeani/dhprebn
	# bndiff = hprebn - bnmeani
	# dbndiff/dhprebn = 1.0
	#
	# bnmeani = 1 / n * hprebn.sum(0, keepdim=True) (1,64) = 1/n (32,64).sum(0)
	# bnmeani is (1, 64), so need to cast to (32,64)
	d_hprebn = (1.0) * d_bndiff
	d_hprebn += (1.0 / n) * torch.ones_like(hprebn) * d_bnmeani
	cmp('hprebn', d_hprebn, hprebn)

	# hprebn = embcat @ W1 + b1
	# hprebn(32, 64) = embcat(32,30)@W1(30,64) + (64)
	# dLoss/dembcat(32,30) = dLoss/dhprebn(32,64) * dhprebn/dembcat(64,30)
	d_embcat = d_hprebn @ W1.T
	cmp('embcat', d_embcat, embcat)

	# dLoss/W1(30,64) =  dhprebn/dW1(30,32) @ dLoss/dhprebn(32,64)
	d_W1 = embcat.T @ d_hprebn
	cmp('W1', d_W1, W1)

	# dLoss/db1(64) = dLoss/dhprebn(32,64) * dhprebn/db1(64)
	# dhprebn/db1 = 1.0
	d_b1 = d_hprebn.sum(0)
	cmp('b1', d_b1, b1)

	# dLoss/demb (32, 3, 10) = dLoss/dembcat(32, 30) * dembcat/demb(32, 3, 10)
	# embcat = emb.view(emb.shape[0], -1)
	d_emb = d_embcat.view(emb.shape)
	cmp('emb', d_emb, emb)

	# emb = C[Xb]
	# emb(32, 3, 10)
	# C(27, 10)
	# Xb(32, 3)
	# Xb will have chunks of 3 characters (1, 1, 4)
	# each integer will specify which row of C to get the 10 features
	# emb is 32 examples by 3 chars by 10 features
	# dLoss/dC (27,10) = dLoss/demb (32, 3, 10) * demb/dCs ()

	dC = torch.zeros_like(C) # (27x10)

	# Iterate through elements at Xb (32, 3)
	for k in range(X.shape[0]):
		for j in range(X.shape[1]):
			ix = X[k, j]
			# forward: took row of C at ix and deposited it at emb at [k, j]
			dC[ix] += d_emb[k, j]	# have to += since the same rows can be used multiple times.

	cmp('C', dC, C)


def fast_backprop(params, parameters, X, Y):

	n = 32
	logits, h, hpreact, bnraw, bnvar_inv, bnvar, hprebn, bnmean, embcat, emb = [params[i] for i in range(len(params))]

	C, W1, b1, W2, b2, bngain, bnbias = [parameters[i] for i in range(len(parameters))]

	# D_logits
	d_logits = F.softmax(logits, 1)	# softmax along rows
	d_logits[range(n), Y] -= 1
	d_logits /= n

	# 2nd layer
	d_h = d_logits @ W2.T
	d_W2 = h.T @ d_logits
	d_b2 = d_logits.sum(0)

	# tanh
	d_hpreact = (1.0 - h**2) * d_h

	# batchnorm
	d_bngain = (bnraw * d_hpreact).sum(0, keepdim=True)	# (32,64)*(32,64) should be a (1,64) --> sum across examples
	d_bnraw = bngain * d_hpreact		# (32,64) = (1,64)*(32,64)
	d_bnbias = (1.0 * d_hpreact).sum(0, keepdim=True)	# (1, 64) = 1.0 * (32,64)

	d_hprebn = bngain*bnvar_inv / n * (n * d_hpreact - d_hpreact.sum(0) - n/(n-1) * bnraw*(d_hpreact*bnraw).sum(0))

	# 1st layer
	d_embcat = d_hprebn @ W1.T
	d_W1 = embcat.T @ d_hprebn
	d_b1 = d_hprebn.sum(0)

	# embedding
	d_emb = d_embcat.view(emb.shape)
	d_C = torch.zeros_like(C)
	for k in range(X.shape[0]):
		for j in range(X.shape[1]):
			ix = X[k, j]
			d_C[ix] += d_emb[k, j]

	grads = [d_C, d_W1, d_b1, d_W2, d_b2, d_bngain, d_bnbias]

	return grads


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

def sample(n_samples, block_size, parameters, seeded=False):

	for i in range(n_samples):
		out = []
		context = [0]*block_size
		itos = get_itos(words)

		while True:

			X = torch.tensor([context])
			#print(X)
			fast_params, logits = fast_forward(X, parameters, sample=True)
			#print(logits)
			probs = F.softmax(logits, dim=1)

			if seeded:
				ix = torch.multinomial(probs, num_samples=1, generator=g).item()
			else:
				ix = torch.multinomial(probs, num_samples=1).item()
			context = context[1:] + [ix]
			out.append(ix)

			if ix==0:
				break
		print(''.join(itos[i] for i in out))

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
	
	N_steps = 10000

	for i in range(N_steps):

		ix = torch.randint(0, X_tr.shape[0], (batch_size, ), generator=g)
		Xb, Yb = X_tr[ix], Y_tr[ix]

		#loss, logits, params = forward_bn(Xb, parameters, Yb)
		fast_params, logits = fast_forward(Xb, parameters)

		loss = fast_loss(logits, Yb)
		
		#parameters, params = backward(loss, parameters, params)
		#backprop(params, parameters, Xb, Yb)

		for p in parameters:
			p.grad = None
		
		grads = fast_backprop(fast_params, parameters, Xb, Yb)

		lr = 0.1

		for p, grad in zip(parameters, grads):
			p.data += -lr * grad

		if i % (N_steps/10) == 0:
			print(f'{i:7d}/{N_steps:7d}: {loss.item():.4f}')

	fast_params, logits = fast_forward(X_dev, parameters)
	loss = fast_loss(logits, Y_dev)
	print("Dev Loss: ", loss.item())

	sample(10, 3, parameters, seeded=False)

main()

'''
clown = torch.tensor([[-1.263600,  0.32630, -0.91806]])
print(clown)
std = clown.std(0, keepdim=True)
print(std)
print(std[0, 0].item() != std[0, 0].item())
'''


