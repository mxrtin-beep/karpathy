

import torch
import torch.nn as nn
from torch.nn import functional as F

def get_text(filename):
	with open(filename, 'r', encoding='utf-8') as f:
		text = f.read()
	return text


text = get_text('input.txt')

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]	# take a string, output a list of ints
decode = lambda l: ''.join([itos[i] for i in l])

device = 'cuda' if torch.cuda.is_available() else 'cpu'




def split(data):

	n = int(0.9 * len(data))
	train_data = data[:n]
	val_data = data[n:]

	return train_data, val_data

def get_batch(split_set, full_data, block_size=8, batch_size=4):

	train_data, val_data = split(full_data)

	data = train_data if split_set=='train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size, ))	# Get 4 random ints anywhere in the dataset: (4, 1) 
	x = torch.stack([data[i:i+block_size] for i in ix])			# Gets the chunks of 8 chars (size block_size), stacks 4 arrays --> (4, 8)
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])		# Gets the chunk of 8 chars immediately following x
	
	x, y = x.to(device), y.to(device)

	return x, y




class Head(nn.Module):
	''' One head of self-attention '''


	def __init__(self, head_size, n_embd, block_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)		# (B, T, C)
		q = self.query(x)	# (B, T, C)

		# Attention scores (affinities)
		wei = q @ k.transpose(-2, -1) * C**-0.5 	# (B, T, C) @ (B, C, T) = (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))	# (B, T, T) decoder block
		wei = F.softmax(wei, dim=-1)	# (B, T, T)

		# Weighted aggregation of the values.
		v = self.value(x)	# (B, T, C)
		out = wei @ v 	# (B, T, T) @ (B, T, C) = (B, T, C)
		return out




class BigramLanguageModel(nn.Module):

	def __init__(self, n_embd, block_size):
		super().__init__()

		self.n_embd = n_embd
		self.block_size = block_size
		# Creates a tensor of shape (vocab_size, n_embd)
		# When you call forward with an idx, it will pluck out a row at that index
		# from the embedding table
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)		# (V, C)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)	# each position from 0 block_size-1 will get an embedding
		self.sa_head = Head(n_embd, n_embd, block_size)
		self.lm_head = nn.Linear(n_embd, vocab_size)						# (C, V)

	# Targets needs to be optional if you just want logits.
	def forward(self, idx, targets=None):

		# 4 (B) examples of 8 (T) tokens
		# Each one has 32 dimensions
		B, T = idx.shape
		
		# idx and targets are both (B, T) tensor of integers

		# Creates (B, T, C) of token embeddings
		# B: batch 		(4)  (batch_size)
		# T: time 		(8)	 (block_size)
		# C: channels	(32) (n_embed)
		# Logits are the scores of the next character in the sequence.
		# Predict what comes next based on identity of ONE token.

		tok_emb = self.token_embedding_table(idx) # (B, T, C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device))	# (T, C)
		#print(torch.arange(T))
		x = tok_emb + pos_emb 		# (B, T, C)
		x = self.sa_head(x)			# (B, T, C) apply one head of self-attention
		logits = self.lm_head(x)	# (B, T, V)

		if targets is None:
			loss = None
		else:
			# Stretch out array such that it's two dimensional, has channels on other side.
			# cross_entropy wants C in last place.
			B, T, C = logits.shape
			logits = logits.view(B*T, C)

			targets = targets.view(B*T)

			# Measures quality of the logits with respect to the targets.
			# Correct dimension of logits should have a very high number and others should be 0.
			loss = F.cross_entropy(logits, targets)

		return logits, loss



	def generate(self, idx, max_new_tokens):
		# idx is a (B, T) array of indices in the current context

		for _ in range(max_new_tokens):

			# With self-attention, can never have more than
			# block_size coming in.
			# Get the last block_size tokens.
			idx_cond = idx[:, -self.block_size:]


			logits, loss = self(idx_cond)

			# Focus only on last time step.
			logits = logits[:, -1, :]	# Becomes (B, C) from (B, T, C), plucks out last T.

			# Apply softmax to get probabilities.
			probs = F.softmax(logits, dim=-1) # (B, C)

			# Sample to get the next character.
			idx_next = torch.multinomial(probs, num_samples=1)	# (B, 1)

			# Append new character to idx.
			idx = torch.cat((idx, idx_next), dim=1)	# (B, T+1)

		return idx

	def sample(self, max_new_tokens):

		context = torch.zeros((1, 1), dtype=torch.long, device=device)

		gen_tensor = self.generate(context, max_new_tokens=max_new_tokens)
		gen_list = gen_tensor[0].tolist() # Index at 0 to pluck out Batch dimension.
		return decode(gen_list)

	def optimize(self, n_steps, batch_size, data):

		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

		for i in range(n_steps):

			if i % (n_steps/10) == 0:
				losses = self.estimate_loss(data, eval_iters=200)
				train_loss = losses['train']
				val_loss = losses['val']
				print(f'Step: {i}; train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')

			xb, yb = get_batch('train', data)

			logits, loss = self(xb, yb)
			optimizer.zero_grad(set_to_none = True)
			loss.backward()
			optimizer.step()




	@torch.no_grad() # Faster, because not storing gradients for backprop.
	def estimate_loss(self, data, eval_iters):

		out = {}
		self.eval()	# Set to evaluation mode. Some layers like BatchNorm act differently.

		for split in ['train', 'val']:
			losses = torch.zeros(eval_iters)
			for k in range(eval_iters):
				X, Y = get_batch(split, data)
				logits, loss = self(X, Y)
				losses[k] = loss.item()
			out[split] = losses.mean()

		self.train() # Set back to training mode.
		return out








# X is a (B, T, C) matrix
# Get a matrix that is the same shape, but each row
# is an average of all the previous characters.

# Wei (T, T) @ (B, T, C) --> (B, T, T) @ (B, T, C) --> (B, T, C)
def get_xbows(X):

	B = X.shape[0]
	T = X.shape[1]
	C = X.shape[2]

	# One head performing self-attention.
	# Each token will have 2 dimensions that it gives:
	# 	key: what it is.
	# 	query: what it wants.
	# Will compute the dot product of one key with all the queries.
	# If key and query are aligned, they will have a high dot product.
	head_size = 16
	key = nn.Linear(C, head_size, bias=False)	# (32 in, 16 out)
	query = nn.Linear(C, head_size, bias=False)	# (32 in, 16 out)
	value = nn.Linear(C, head_size, bias=False)	# (32 in, 16 out)

	# Converts C (32 dimensions of each char) to 16 keys and queries.
	k = key(X)		# (B, T, 16)
	q = query(X)	# (B, T, 16)

	# Affinities
	wei = q @ k.transpose(-2, -1) * head_size**-0.5	# (B, T, 16) @ (B, 16, T) --> (B, T, T)
	# For every row of B (for every batch), you will have a TxT array of affinities.
	# Wei: (B, T, T) random array of weights.s
	# Divide by squareroot of head_size to keep the variance 1.
	# Otherwise, if you have very positive and negative numbers, softmax will converge to one hot encoding,
	# and you will only get info from the largest node.

	# Triangular matrix
	# Bottom triangle is ones, upper triangle is zeroes.
	tril = torch.tril(torch.ones(T, T))

	# Triangular matrix (weights).
	# Bottom triangle is zeroes, upper triangle is -inf.
	# Now, weigh is lower triangular.
	# If you remove this line, all the characters will communicate with each other.
	# ENCODER blocks don't have this, DECODER blocks do.
	wei = wei.masked_fill(tril == 0, float('-inf'))	# Tokens from the past cannot communicate.

	# Exponentiate (0 --> 1, -inf --> 0)
	# Take the average (first row: 1; second row: 0.5, 0.5; third row: 0.33, 0.33, 0.33, etc.)
	wei = F.softmax(wei, dim=-1)
	# Now, all rows of wei sum to 1.

	# v is the thing that gets communicated if a token is found interesting.
	# v is the thing that gets aggregated.
	v = value(X)	# (B, T, C) --> (B, T, 16)

	# Can imagine a directed graph of nodes.
	# Every node has a vector of info; can aggregate info from a weighted sum of
	# all the nodes pointing to it.
	# We have 8 nodes (batch_size).
	#	First node is only pointed to by itself.
	#	Second node is only pointed to by itself and the first node.
	# 	etc.
	#	Eigth node is pointed to by all the nodes.
	# NO NOTION OF SPACE; that's why you encode space.
	# Also, each batch is independent from each other of course.
	# Future tokens don't communicate to the past tokens, but that is specific to this.
	# In many cases you might want all of the tokens talk to each other, for example
	# if you want to get the sentiment of a sentence for example.

	out = wei @ v
	return out


	# Wei will have a T, T array for each 8-char word in the batch.
	# The array will be a lower triangular array.
	# Instead of have them all be 
	#	[1.0, 0.0, 0.0]
	#	[0.5, 0.5, 0.0]
	#	[0.3, 0.3, 0.3]

	# Each of them will be different (initialized randomly) and normalized to sum to 1 again.
	# For example: last token of last row:
	# 	Knows what it is (vowel), knows what position it is --> generates query
	#	"hey i'm a vowel, i'm looking for any consonants in positions up to 4"
	#	every channel emits a key (i'm a consonant at position 4) --> that key would have a
	# 	high number in that specific channel (8th row, 4th position)
	# 	--> high affinity when they dot product
	# 	when they have a high affinity, when you softmax, you add a larger portion of its information
	# 	to your information, and you will learn about it.

	# This is SELF-ATTENTION, because the same source x produces keys, queries, and values.
	# Can have CROSS-ATTENTION where that is not the case.

	# In General: Attention(Q, K, V) = softmax((QK.T) / Sqrt(d_k)) *V

def main():

	torch.manual_seed(1389)

	data = torch.tensor(encode(text), dtype=torch.long)
	print(data.shape, data.dtype)
	

	block_size = 8		# Maximum size of training chunk
	batch_size = 32 	# How many independent sequences will we process in parallel?
	n_embd = 32


	# If you have a block_size of 8, you have 8 examples packed into one.
	# For example, if your block is [18, 47, 56, 57, 1, ...], 
	# Your examples are 18 --> 47; (18, 47) --> 56; (18, 47, 56) --> 57, and so on

	xb, yb = get_batch('train', data)

	model = BigramLanguageModel(n_embd, block_size)
	m = model.to(device)

	logits, loss = m(xb, yb)
	print(f'Loss: {loss}')

	# Batch is 1, Time is 1, holds a 0 (newLine character).
	#print(m.sample(100))

	m.optimize(10000, batch_size=batch_size, data=data)
	#print(m.estimate_loss(data, eval_iters=200))

	print(m.sample(400))

main()
