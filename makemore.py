

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
	N = torch.zeros((27, 27), dtype=torch.int32)

	# Concat the entire document to one string, throw out duplicates
	# Make it a list and sort it
	chars = sorted(list(set(''.join(words))))

	# Map each char to an integer, add the start and end
	# Add a dot to start and end
	#	Problem with unique start and end chars is that nothing will ever
	#	follow <E>, and nothing will ever be followed by <S>
	# Solution: one special char.
	stoi = {s:i+1 for i,s in enumerate(chars)}
	stoi['.'] = 0

	itos = {i:s for s,i in stoi.items()}

	for w in words:

		chs = ['.'] + list(w) + ['.']
		for ch1, ch2 in zip(chs, chs[1:]):
			ix1 = stoi[ch1]
			ix2 = stoi[ch2]
			N[ix1, ix2] += 1


	if plot_arr:
		plt.figure(figsize=(16,16))
		plt.imshow(N, cmap='Blues')

		for i in range(27):
			for j in range(27):
				chstr = itos[i] + itos[j]
				plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
				plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
		plt.axis('off')

		plt.savefig('freq.png')

	return N


def get_stoi(words):
	chars = sorted(list(set(''.join(words))))
	stoi = {s:i+1 for i,s in enumerate(chars)}
	stoi['.'] = 0

	return stoi

def get_itos(words):
	stoi = get_stoi(words)
	itos = {i:s for s,i in stoi.items()}
	return itos


N = get_bigram_matrix(words)

def normalize_row(N, row):	
	# N[0, :] or N[0] will give you the first row
	# Want to sample from it; convert to probabilities

	p = N[row].float()
	p /= p.sum()
	#print(p)
	return p

# p is a normalized row
def sample(p, words):

	g = torch.Generator().manual_seed(2147483647)
	ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
	char = get_itos(words)[ix]
	
	#print(ix)
	return ix


def sample_words(words, samples):

	N = get_bigram_matrix(words)

	g = torch.Generator().manual_seed(2147483647)

	for i in range(samples):
		out = []
		ix = 0

		while True:

			p = normalize_row(N, ix)
			ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

			out.append(get_itos(words)[ix])
			if ix == 0: # end
				break

		print(''.join(out))
	#return out

sample_words(words, 10)