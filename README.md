## How to Run

The entire project is controlled through a single entry point, `main.py`. Pass a part argument to select which experiment to run.

## Summary

For Part 1 , run :
python main.py 

For Part 2 , run :
python main.py part2

For Part 3 , run :
python main.py part3


### Part 1 — Encoder + Classifier (default)

Trains a `TransformerEncoder` end-to-end with a feedforward classifier to identify which of three politicians (Obama, W. Bush, H. Bush) gave a speech. Trains for 15 epochs and reports train/test accuracy each epoch.

```bash
python main.py
```

**What it does:**
- Builds a word-level vocabulary from all training texts
- Trains `TransformerEncoder` + `FeedforwardClassifier` jointly using cross-entropy loss
- Reports train accuracy, test accuracy, and final parameter count


### Part 2 — Decoder + Language Modeling

Trains a `TransformerDecoder` with causal (masked) self-attention for next-token prediction. Evaluates perplexity on training data and on each of the three politicians' test sets.

```bash
python main.py part2
```

**What it does:**
- Trains a decoder-only transformer for up to 500 iterations
- Evaluates train and test perplexity every 100 iterations
- Reports final perplexity separately for Obama, W. Bush, and H. Bush test sets

### Part 3 — Architectural Exploration (ALiBi + Local Window Attention)

Trains the same classification task as Part 1, but swaps in `ALiBiTransformerEncoder` — a modified encoder that replaces learned positional embeddings with **ALiBi (Attention with Linear Biases)** and optionally restricts attention to a **local window**.

```bash
python main.py part3
```
## Attention Sanity Check
Visualizes attention maps for two example sentences. Saves heatmap images (attention_map_1.png, attention_map_2.png, etc.) to the working directory.

For part 1 (default encoder): 
python attention_sanity_check.py

For part 3 (custom encoder):
python attention_sanity_check.py part 3

For part 2 :
python decoder_attention_sanity_check.py


## Model Architecture

### TransformerEncoder (Parts 1 & 2)

| Component | Details |
|---|---|
| Token embedding | `nn.Embedding(vocab_size, 64)` |
| Positional embedding | `nn.Embedding(32, 64)` — learned |
| Encoder layers | 4 × `TransformerEncoderLayer` |
| Attention heads | 2 |
| FFN hidden dim | 256 (4 × n_embd) |
| Activation | GELU |
| Pooling | Masked mean pooling |

### FeedforwardClassifier

| Component | Details |
|---|---|
| Input | 64 (matches encoder embedding dim) |
| Hidden layer | 100 units + ReLU |
| Output | 3 classes |

### TransformerDecoder (Part 2)

Same embedding size and depth as the encoder, but uses causal self-attention masking (upper-triangular `-inf` mask) to prevent attending to future tokens. An all-zeros memory tensor is used in place of encoder cross-attention output.

### ALiBiTransformerEncoder (Part 3)

| Change | Description |
|---|---|
| No `pos_emb` | Positional embedding layer removed entirely |
| ALiBi bias | Each head gets a slope `m_h = 2^(-8h/n_heads)`; attention logit penalized by `-m_h × \|i−j\|` |
| Local window (optional) | Set `window=8` in the constructor to restrict each token to attending only ±4 neighbors |

To switch to local-window attention in Part 3, edit line 206 of `main.py`:

```python
# Full attention (default)
encoder = ALiBiTransformerEncoder(..., window=None)

# Local window of width 8
encoder = ALiBiTransformerEncoder(..., window=8)
```
