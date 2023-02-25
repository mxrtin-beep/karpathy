
import torch
import matplotlib.pyplot as plt


words = open('names.txt', 'r').read().splitlines()

# Training set of all bigrams.

def get_stoi(words):
	chars = sorted(list(set(''.join(words))))
	stoi = {s:i+1 for i,s in enumerate(chars)}
	stoi['.'] = 0

	return stoi

def get_itos(words):
	stoi = get_stoi(words)
	itos = {i:s for s,i in stoi.items()}
	return itos

# Create training set of bigrams (x, y)

def get_training_set(words, n_words):
	xs, ys = [], []
	stoi = get_stoi(words)

	for w in words[:n_words]:

		chs = ['.'] + list(w) + ['.']
		for ch1, ch2 in zip(chs, chs[1:]):
			ix1 = stoi[ch1]
			ix2 = stoi[ch2]
			xs.append(ix1)
			ys.append(ix2)

	# Lowercase tensor finds the dtype implicitly.
	# Uppercase Tensor uses float.
	xs = torch.tensor(xs)
	ys = torch.tensor(ys)
	#print(xs)
	#print(ys)
	return xs, ys

#xs, ys = get_training_set(words, 1)

import torch.nn.functional as F

def get_probs(words, n_words, W):
	
	#print("Getting Probabilities")
	xs, ys = get_training_set(words, n_words)

	xenc = F.one_hot(xs, num_classes=27).float()

	# 27 x n neurons
	# 27 neurons (second parameter)
	g = torch.Generator().manual_seed(2147483647)
	

	# Matrix Multiplication
	logits = xenc @ W
	# 5x1 array
	# 5x27 @ 27x1 = 5x1

	# 5x27 @ 27x27 = 27x27
	# xenc @ W [a, b] gives firing rate of bth neuron on ath input
	# dot product between ath input and bth column

	# How to understand the 27 outputs?
	# Not probabilities, because they're negative and positive and don't sum to 1.
	# Not counts, because not positive integers.
	# Answer: they're giving us log-counts. Exponentiate to get counts.
	# Exponentiate to make negative numbers less than 1 and positive numbers greater than 1.

	# Softmax
	counts = logits.exp()
	probs = counts / counts.sum(1, keepdims=True)

	#print(probs[0].sum())
	#print(probs.shape)
	return probs


def evaluate(words, n_words, n_epochs, verbose = False):

	g = torch.Generator().manual_seed(2147483647)
	W = torch.randn((27, 27), generator=g, requires_grad=True)

	xs, ys = get_training_set(words, n_words)
	print(xs)

	nlls = torch.zeros(n_words)

	itos = get_itos(words)

	for i in range(n_epochs):
		# Making predictions
		probs = get_probs(words, n_words, W) # y_pred

		# Getting the loss
		for i in range(n_words):

			# i-th bigram
			x = xs[i].item() 	# input char index
			y = ys[i].item()	# label char index

			if verbose:
				print('--------')
				print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indices {x}{y})')
				print('input to the neural net: ', x)
				print('output probabilities from the neural net:', probs[i])
				print('label (actual next character): ', y)
			

			p = probs[i, y]

			if verbose: print('probability assigned by the net to the correct character: ', p.item())
			logp = torch.log(p)

			if verbose: print('log likelihood:', logp.item())
			nll = -logp

			if verbose: print('negative log likelihood: ', nll.item())
			nlls[i] = nll

		print('===========')
		print('average nll, loss =', nlls.mean().item())

		# Probabilities of next character
		ps_next_char = probs[torch.arange(n_words), ys]
		loss = -ps_next_char.log().mean()

		#return loss

		# backward pass
		W.grad = None

		loss.backward()

		W.data += -50 * W.grad

#nlls = evaluate(words, 228146, 228146, 50)

# Probs is our y_pred
# evaluating it based on nll instead of mse as the loss

# probs.shape is 5, 27
# looking for probs[0, 5], probs[1, 13], probs[2, 13], probs[3, 1], probs[4, 0]
# (probabilities of correct labels)
# can create first index with torch.arange(5), ys



# ------------------------------------------------------------------------------------------------

def forward(xs, ys, W, num, regularize, return_p = False):

	reg_loss = 0.01

	xenc = F.one_hot(xs, num_classes=27).float()
	logits = xenc @ W
	counts = logits.exp()
	probs = counts / counts.sum(1, keepdims=True)
	loss = -probs[torch.arange(num), ys].log().mean()

	# Tries to make weights go to zero.
	if regularize:
		loss += reg_loss * (W**2).mean()


	if return_p: return probs

	return loss

def backward(W, loss, learning_rate = 0.1):
	W.grad = None
	loss.backward()

	W.data += -1 * learning_rate * W.grad

	return W


def short_evaluate(words, n_epochs, n_examples, regularize = False):
	g = torch.Generator().manual_seed(2147483647)
	W = torch.randn((27, 27), generator=g, requires_grad=True)

	xs, ys = get_training_set(words, n_examples)

	for i in range(n_epochs):

		loss = forward(xs, ys, W, n_examples, regularize)

		print('Loss: ', loss.item())

		W = backward(W, loss, 50)

	return W

#short_evaluate(words, 100, 228146, regularize=True)



def sample(words, W):

	g = torch.Generator().manual_seed(2147483647+1)
	itos = get_itos(words)

	for i in range(5):
		out = []
		ix = 0

		while True:
			xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
			logits = xenc @ W # Plucks out row of W corresponding to ix

			counts = logits.exp()
			probs = counts / counts.sum(1, keepdims=True)

			ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
			out.append(itos[ix])

			if ix==0:
				break
		print(''.join(out))

W = short_evaluate(words, 10, 228146, regularize=True)


sample(words, W)
