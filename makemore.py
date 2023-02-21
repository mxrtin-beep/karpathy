

import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

#print(words[:10])


# Bigrams

def get_bigram_dict(words):
	b = {}

	for w in words:

		chs = ['<S>'] + list(w) + ['<E>']
		for ch1, ch2 in zip(chs, chs[1:]):

			bigram = (ch1, ch2)
			b[bigram] = b.get(bigram, 0) + 1
			#print(ch1, ch2)

	return b

#print(b)

# Sort it in increasing order
#print(sorted(b.items(), key = lambda kv: -kv[1]))

# Store it in a 2D array with first letter as rows, second letter as cols


def get_bigram_matrix(words, plot_arr=True):
	N = torch.zeros((28, 28), dtype=torch.int32)

	# Concat the entire document to one string, throw out duplicates
	# Make it a list and sort it
	chars = sorted(list(set(''.join(words))))

	# Map each char to an integer, add the start and end
	stoi = {s:i for i,s in enumerate(chars)}
	stoi['<S>'] = 26
	stoi['<E>'] = 27

	itos = {i:s for s,i in stoi.items()}

	for w in words:

		chs = ['<S>'] + list(w) + ['<E>']
		for ch1, ch2 in zip(chs, chs[1:]):
			ix1 = stoi[ch1]
			ix2 = stoi[ch2]
			N[ix1, ix2] += 1


	if plot_arr:
		plt.figure(figsize=(16,16))
		plt.imshow(N, cmap='Blues')

		for i in range(28):
			for j in range(28):
				chstr = itos[i] + itos[j]
				plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
				plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
		plt.axis('off')

		plt.savefig('freq.png')

	return N


N = get_bigram_matrix(words)
	
