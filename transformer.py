
# Implementation of a Transformer Encoder and Decoder using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (should match those in main.py)
n_embd = 64  # Embedding dimension
n_head = 2   # Number of attention heads
n_layer = 4  # Number of transformer layers

# Transformer Decoder for Part 2
class TransformerDecoder(nn.Module):
	def __init__(self, vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, max_seq_len=32, ff_hidden_dim=100):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, n_embd)
		self.pos_emb = nn.Embedding(max_seq_len, n_embd)
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=n_embd,
			nhead=n_head,
			dim_feedforward=ff_hidden_dim,
			activation='relu',
			batch_first=True
		)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
		self.n_embd = n_embd
		self.max_seq_len = max_seq_len
		self.lm_head = nn.Linear(n_embd, vocab_size)

	def generate_square_subsequent_mask(self, sz):
		# Mask out future tokens (upper triangle)
		mask = torch.triu(torch.ones(sz, sz), diagonal=1)
		mask = mask.masked_fill(mask == 1, float('-inf'))
		return mask

	def forward(self, x, targets=None):
		# x: (batch_size, seq_len)
		batch_size, seq_len = x.size()
		positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
		tok_emb = self.token_emb(x)
		pos_emb = self.pos_emb(positions)
		h = tok_emb + pos_emb
		# Masked self-attention
		tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
		# No encoder input, so use zeros as memory
		memory = torch.zeros(batch_size, seq_len, self.n_embd, device=x.device)
		out = self.transformer_decoder(h, memory, tgt_mask=tgt_mask)
		logits = self.lm_head(out)
		if targets is not None:
			# Shift logits and targets for language modeling
			logits = logits[:, :-1, :].contiguous()
			targets = targets[:, 1:].contiguous()
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
			return loss
		return logits

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

		# Register hooks to capture attention weights
		self.attn_maps = []
		def hook(module, input, output):
			if hasattr(module, 'self_attn'):
				attn_output, attn_weights = module.self_attn(output, output, output)
				self.attn_maps.append(attn_weights.detach())
		for layer in self.transformer_encoder.layers:
			layer.register_forward_hook(hook)

	def forward(self, x):
		# x: (batch_size, seq_len)
		self.attn_maps = []  # Clear previous attention maps
		batch_size, seq_len = x.size()
		positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
		tok_emb = self.token_emb(x)  # (batch_size, seq_len, n_embd)
		pos_emb = self.pos_emb(positions)  # (batch_size, seq_len, n_embd)
		h = tok_emb + pos_emb
		out = self.transformer_encoder(h)  # (batch_size, seq_len, n_embd)
		return out, self.attn_maps

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
