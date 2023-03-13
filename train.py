

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


class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()

		# Creates a tensor of shape (vocab_size, vocab_size)
		# When you call forward with an idx, it will pluck out a row at that index
		# from the embedding table
		self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

	# Targets needs to be optional if you just want logits.
	def forward(self, idx, targets=None):

		# idx and targets are both (B, T) tensor of integers

		# Creates (B, T, C)
		# B: batch 		(4)
		# T: time 		(8)
		# C: channels	(65) (vocab_size)
		# Logits are the scores of the next character in the sequence.
		# Predict what comes next based on identity of ONE token.
		logits = self.token_embedding_table(idx)

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

			logits, loss = self(idx)

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
		#print(gen_tensor.shape)
		gen_list = gen_tensor[0].tolist() # Index at 0 to pluck out Batch dimension.
		return decode(gen_list)

	def optimize(self, n_steps, batch_size, data):

		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

		for steps in range(n_steps):

			xb, yb = get_batch('train', data)

			logits, loss = self(xb, yb)
			optimizer.zero_grad(set_to_none = True)
			loss.backward()
			optimizer.step()

		print(loss.item())

def main():

	torch.manual_seed(1389)

	data = torch.tensor(encode(text), dtype=torch.long)
	print(data.shape, data.dtype)
	

	block_size = 8		# Maximum size of training chunk
	batch_size = 32 	# How many independent sequences will we process in parallel?


	# If you have a block_size of 8, you have 8 examples packed into one.
	# For example, if your block is [18, 47, 56, 57, 1, ...], 
	# Your examples are 18 --> 47; (18, 47) --> 56; (18, 47, 56) --> 57, and so on

	xb, yb = get_batch('train', data)

	model = BigramLanguageModel(vocab_size)
	m = model.to(device)

	logits, loss = m(xb, yb)
	print(logits.shape)
	print(f'Loss: {loss}')

	# Batch is 1, Time is 1, holds a 0 (newLine character).
	print(m.sample(100))

	m.optimize(10000, batch_size=batch_size, data=data)

	print(m.sample(100))
main()
