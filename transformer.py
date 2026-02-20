# Implementation of a Transformer Encoder and Decoder using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (should match those in main.py)
n_embd = 64  # Embedding dimension
n_head = 2   # Number of attention heads
n_layer = 4  # Number of transformer layers

# Transformer Decoder for Part 2

# Custom Decoder Layer to extract attention weights
class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.self_attn_weights = None

	def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
				tgt_key_padding_mask=None, memory_key_padding_mask=None):
		# Self-attention
		tgt2, attn_weights = self.self_attn(
			tgt, tgt, tgt,
			attn_mask=tgt_mask,
			key_padding_mask=tgt_key_padding_mask,
			need_weights=True,
			average_attn_weights=False
		)
		self.self_attn_weights = attn_weights  # (batch, num_heads, seq, seq)
		tgt = tgt + self.dropout1(tgt2)
		tgt = self.norm1(tgt)
		# Cross-attention (not used here, memory is zeros)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout2(tgt2)
		tgt = self.norm2(tgt)
		return tgt

class TransformerDecoder(nn.Module):
	def __init__(self, vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, max_seq_len=32, ff_hidden_dim=100):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, n_embd)
		self.pos_emb = nn.Embedding(max_seq_len, n_embd)
		decoder_layers = nn.ModuleList([
			CustomTransformerDecoderLayer(
				d_model=n_embd,
				nhead=n_head,
				dim_feedforward=ff_hidden_dim,
				activation='relu',
				batch_first=True
			) for _ in range(n_layer)
		])
		self.decoder_layers = decoder_layers
		self.n_embd = n_embd
		self.max_seq_len = max_seq_len
		self.lm_head = nn.Linear(n_embd, vocab_size)

	def generate_square_subsequent_mask(self, sz):
		# Mask out future tokens (upper triangle)
		mask = torch.triu(torch.ones(sz, sz), diagonal=1)
		mask = mask.masked_fill(mask == 1, float('-inf'))
		return mask

	def forward(self, x, targets=None, return_attn=False):
		# x: (batch_size, seq_len)
		attn_maps = []
		batch_size, seq_len = x.size()
		positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
		tok_emb = self.token_emb(x)
		pos_emb = self.pos_emb(positions)
		h = tok_emb + pos_emb
		tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
		memory = torch.zeros(batch_size, seq_len, self.n_embd, device=x.device)
		for layer in self.decoder_layers:
			h = layer(h, memory, tgt_mask=tgt_mask)
			if hasattr(layer, 'self_attn_weights') and layer.self_attn_weights is not None:
				attn_maps.append(layer.self_attn_weights.detach().cpu())
		logits = self.lm_head(h)
		if targets is not None:
			logits = logits[:, :-1, :].contiguous()
			targets = targets[:, 1:].contiguous()
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
			if return_attn:
				return loss, attn_maps
			return loss
		if return_attn:
			return logits, attn_maps
		return logits

# Hyperparameters (should match those in main.py)
n_embd = 64  # Embedding dimension
n_head = 2   # Number of attention heads
n_layer = 4  # Number of transformer layers

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.attn_weights = None

	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		# src: (batch, seq, d_model)
		src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=True, average_attn_weights=False)
		self.attn_weights = attn_weights  # (batch, num_heads, seq, seq)
		src = src + self.dropout1(src2)
		src = self.norm1(src)
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)
		return src

class TransformerEncoder(nn.Module):
	def __init__(self, vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, max_seq_len=32):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, n_embd)
		self.pos_emb = nn.Embedding(max_seq_len, n_embd)
		encoder_layers = nn.ModuleList([
			CustomTransformerEncoderLayer(
				d_model=n_embd,
				nhead=n_head,
				dim_feedforward=4 * n_embd,
				activation='gelu',
				batch_first=True
			) for _ in range(n_layer)
		])
		self.encoder_layers = encoder_layers
		self.n_embd = n_embd
		self.max_seq_len = max_seq_len

	def forward(self, x):
		# x: (batch_size, seq_len)
		attn_maps = []
		batch_size, seq_len = x.size()
		positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
		tok_emb = self.token_emb(x)  # (batch_size, seq_len, n_embd)
		pos_emb = self.pos_emb(positions)  # (batch_size, seq_len, n_embd)
		h = tok_emb + pos_emb
		for layer in self.encoder_layers:
			h = layer(h)
			if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
				attn_maps.append(layer.attn_weights.detach().cpu())
		return h, attn_maps

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


# =============================================================================
# Part 3: Architectural Exploration — ALiBi + Local Window Attention
# =============================================================================
import math

def _get_alibi_slopes(n_heads):
	"""Compute head-specific ALiBi slopes (Press et al., 2022)."""
	def _pow2_slopes(n):
		start = 2 ** (-(2 ** -(math.log2(n) - 3)))
		return [start * (start ** i) for i in range(n)]

	if math.log2(n_heads) % 1 == 0:
		return torch.tensor(_pow2_slopes(n_heads))
	closest = 2 ** math.floor(math.log2(n_heads))
	base  = _pow2_slopes(closest)
	extra = _pow2_slopes(2 * closest)[0::2][: n_heads - closest]
	return torch.tensor(base + extra)


def _alibi_bias(seq_len, n_heads, device):
	"""Returns ALiBi bias of shape (n_heads, seq_len, seq_len)."""
	slopes = _get_alibi_slopes(n_heads).to(device)        # (H,)
	pos    = torch.arange(seq_len, device=device)
	dist   = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
	return -slopes.view(n_heads, 1, 1) * dist.float()     # (H, T, T)


def _window_mask(seq_len, window, device):
	"""Additive mask: 0 inside window, -inf outside."""
	pos  = torch.arange(seq_len, device=device)
	dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
	mask = torch.zeros(seq_len, seq_len, device=device)
	mask[dist > window // 2] = float('-inf')
	return mask                                            # (T, T)


class _ALiBiMHA(nn.Module):
	"""Multi-head attention with ALiBi bias and optional local-window mask."""
	def __init__(self, d_model, n_heads, window=None, dropout=0.1):
		super().__init__()
		self.H, self.d_k = n_heads, d_model // n_heads
		self.window = window
		self.W_q = nn.Linear(d_model, d_model, bias=False)
		self.W_k = nn.Linear(d_model, d_model, bias=False)
		self.W_v = nn.Linear(d_model, d_model, bias=False)
		self.W_o = nn.Linear(d_model, d_model, bias=False)
		self.drop = nn.Dropout(dropout)

	def forward(self, x, pad_mask=None):
		B, T, D = x.shape
		H, d_k  = self.H, self.d_k
		def proj(W): return W(x).view(B, T, H, d_k).transpose(1, 2)
		Q, K, V = proj(self.W_q), proj(self.W_k), proj(self.W_v)

		scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)           # (B,H,T,T)
		scores = scores + _alibi_bias(T, H, x.device).unsqueeze(0)     # ALiBi
		if self.window is not None:
			scores = scores + _window_mask(T, self.window, x.device)   # local window
		if pad_mask is not None:
			scores = scores.masked_fill(pad_mask[:, None, None, :], float('-inf'))

		attn = self.drop(torch.softmax(scores, dim=-1))
		out  = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
		return self.W_o(out), attn.detach()


class _ALiBiLayer(nn.Module):
	def __init__(self, d_model, n_heads, ff_dim, window=None, dropout=0.1):
		super().__init__()
		self.attn  = _ALiBiMHA(d_model, n_heads, window=window, dropout=dropout)
		self.ff    = nn.Sequential(
			nn.Linear(d_model, ff_dim), nn.GELU(),
			nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
		)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.drop  = nn.Dropout(dropout)

	def forward(self, x, pad_mask=None):
		a, w = self.attn(x, pad_mask=pad_mask)
		x = self.norm1(x + self.drop(a))
		x = self.norm2(x + self.ff(x))
		return x, w


class ALiBiTransformerEncoder(nn.Module):
	"""
	Drop-in replacement for TransformerEncoder using:
	  • ALiBi positional bias  (no learned pos embedding)
	  • Optional local-window attention  (window kwarg, None = full attention)

	API identical to TransformerEncoder: forward(x) -> (out, attn_maps)
	                                     mean_pool(x, mask) -> pooled
	"""
	def __init__(self, vocab_size, n_embd=64, n_head=2, n_layer=4,
	             max_seq_len=32, window=None, dropout=0.1):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, n_embd)   # no pos_emb!
		self.layers    = nn.ModuleList([
			_ALiBiLayer(n_embd, n_head, 4 * n_embd, window=window, dropout=dropout)
			for _ in range(n_layer)
		])
		self.norm   = nn.LayerNorm(n_embd)
		self.n_embd = n_embd
		self.attn_maps = []

	def forward(self, x):
		self.attn_maps = []
		pad_mask = (x == 0)                   # True for <pad> tokens
		h = self.token_emb(x)                 # (B, T, n_embd) — no pos embedding
		for layer in self.layers:
			h, aw = layer(h, pad_mask=pad_mask)
			self.attn_maps.append(aw)
		return self.norm(h), self.attn_maps

	def mean_pool(self, x, mask=None):
		if mask is None:
			return x.mean(dim=1)
		mask = mask.unsqueeze(-1)
		return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)