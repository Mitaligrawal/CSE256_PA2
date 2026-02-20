# Implementation of a Transformer Encoder using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (should match those in main.py)
n_embd = 64  # Embedding dimension
n_head = 2   # Number of attention heads
n_layer = 4  # Number of transformer layers

class TransformerEncoder(nn.Module):
	def __init__(self, vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, max_seq_len=32):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, n_embd)
		self.pos_emb = nn.Embedding(max_seq_len, n_embd)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=n_embd,
			nhead=n_head,
			dim_feedforward=4 * n_embd,
			activation='gelu',
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
		self.n_embd = n_embd
		self.max_seq_len = max_seq_len

	def forward(self, x):
		# x: (batch_size, seq_len)
		batch_size, seq_len = x.size()
		positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
		tok_emb = self.token_emb(x)  # (batch_size, seq_len, n_embd)
		pos_emb = self.pos_emb(positions)  # (batch_size, seq_len, n_embd)
		h = tok_emb + pos_emb
		# Transformer expects (batch_size, seq_len, n_embd) if batch_first=True
		out = self.transformer_encoder(h)  # (batch_size, seq_len, n_embd)
		return out

	def mean_pool(self, x, mask=None):
		# x: (batch_size, seq_len, n_embd)
		# mask: (batch_size, seq_len) with 1 for valid tokens, 0 for padding
		if mask is None:
			return x.mean(dim=1)
		else:
			mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
			summed = (x * mask).sum(dim=1)
			counts = mask.sum(dim=1).clamp(min=1)
			return summed / counts
