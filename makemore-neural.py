
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


	for w in words[:n_words]:

		chs = ['.'] + list(w) + ['.']
		for ch1, ch2 in zip(chs, chs[1:]):
			ix1 = get_stoi(words)[ch1]
			ix2 = get_stoi(words)[ch2]
			xs.append(ix1)
			ys.append(ix2)

	# Lowercase tensor finds the dtype implicitly.
	# Uppercase Tensor uses float.
	xs = torch.tensor(xs)
	ys = torch.tensor(ys)
	#print(xs)
	#print(ys)
	return xs, ys

xs, ys = get_training_set(words, 1)

import torch.nn.functional as F

def train(words, n_words):
	xs, ys = get_training_set(words, n_words)

	xenc = F.one_hot(xs, num_classes=27).float()

	# 27 x n neurons
	# 27 neurons (second parameter)
	g = torch.Generator().manual_seed(2147483647)
	W = torch.randn((27, 27), generator=g)

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

	print(probs[0].sum())
	print(probs.shape)

nlls = torch.zeros(5)
for i in range(5):

	# i-th bigram
	x = xs[i].item()
