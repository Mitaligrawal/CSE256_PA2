# Transformer-Based Speech Classification & Language Modeling

A from-scratch implementation of Transformer Encoder and Decoder models applied to two NLP tasks: speaker classification and language modeling on political speech data.

---

## Project Structure

```
.
├── main.py                   # Entry point — runs Part 1, 2, or 3
├── transformer.py            # TransformerEncoder, TransformerDecoder, ALiBiTransformerEncoder
├── classifier.py             # Feedforward classifier head (1 hidden layer)
├── tokenizer.py              # Simple word-level tokenizer
├── dataset.py                # Dataset classes for classification and LM tasks
├── utilities.py              # Attention map visualization utilities
├── attention_sanity_check.py # Standalone script to visualize attention maps
└── speechesdataset/
    ├── train_CLS.tsv         # Classification training data  (label \t text)
    ├── test_CLS.tsv          # Classification test data
    ├── train_LM.txt          # Language modeling training text
    ├── test_LM_obama.txt     # LM test set — Obama speeches
    ├── test_LM_wbush.txt     # LM test set — W. Bush speeches
    └── test_LM_hbush.txt     # LM test set — H. Bush speeches
```

---

## Requirements

- Python 3.8+
- PyTorch
- NLTK (for tokenization)

Install dependencies:

```bash
pip install torch nltk
python -c "import nltk; nltk.download('punkt')"
```

---

## How to Run

The entire project is controlled through a single entry point, `main.py`. Pass a part argument to select which experiment to run.

### Part 1 — Encoder + Classifier (default)

Trains a `TransformerEncoder` end-to-end with a feedforward classifier to identify which of three politicians (Obama, W. Bush, H. Bush) gave a speech. Trains for 15 epochs and reports train/test accuracy each epoch.

```bash
python main.py
```

**What it does:**
- Builds a word-level vocabulary from all training texts
- Trains `TransformerEncoder` + `FeedforwardClassifier` jointly using cross-entropy loss
- Reports train accuracy, test accuracy, and final parameter count

**Expected output:**
```
Loading data and creating tokenizer ...
Vocabulary size is XXXX
Epoch 1/15 | Loss: 1.0843 | Train Accuracy: 45.23% | Test Accuracy: 42.10%
...
Epoch 15/15 | Loss: 0.3012 | Train Accuracy: 91.40% | Test Accuracy: 78.50%
Final Test Accuracy: 78.50%
Number of parameters in encoder: XXXXX
```

---

### Part 2 — Decoder + Language Modeling

Trains a `TransformerDecoder` with causal (masked) self-attention for next-token prediction. Evaluates perplexity on training data and on each of the three politicians' test sets.

```bash
python main.py part2
```

**What it does:**
- Trains a decoder-only transformer for up to 500 iterations
- Evaluates train and test perplexity every 100 iterations
- Reports final perplexity separately for Obama, W. Bush, and H. Bush test sets

**Expected output:**
```
Running Part 2: Decoder Implementation
Vocabulary size is XXXX
Number of parameters in decoder: XXXXX
Training decoder...
Iter 100: Train Perplexity 245.32
Iter 100: Test Perplexity (Obama): 312.45
...
Final Train Perplexity: 98.21
Final Test Perplexity (Obama): 187.34
Final Test Perplexity (W. Bush): 201.12
Final Test Perplexity (H. Bush): 195.67
```

---

### Part 3 — Architectural Exploration (ALiBi + Local Window Attention)

Trains the same classification task as Part 1, but swaps in `ALiBiTransformerEncoder` — a modified encoder that replaces learned positional embeddings with **ALiBi (Attention with Linear Biases)** and optionally restricts attention to a **local window**.

```bash
python main.py part3
```

**What it does:**
- Uses `ALiBiTransformerEncoder` instead of `TransformerEncoder`
- No learned positional embedding — positions are encoded as per-head distance penalties on attention logits
- Reports parameter count (should be ~2,048 fewer than Part 1 due to removed `pos_emb`)
- Trains and evaluates the classifier identically to Part 1

**Expected output:**
```
Running Part 3: Architectural Exploration (ALiBi + Local Window Attention)
Vocabulary size is XXXX
Number of parameters in ALiBi encoder: XXXXX
Training ALiBi encoder + classifier...
Epoch 1/15 | Loss: 1.0712 | Train Acc: 46.10% | Test Acc: 43.20%
...
Final Test Accuracy (ALiBi): 79.30%
```

---

### Attention Sanity Check

Visualizes attention maps for two example sentences. Saves heatmap images (`attention_map_1.png`, `attention_map_2.png`, etc.) to the working directory.

```bash
python attention_sanity_check.py
```

---

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

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 16 | Sequences processed in parallel |
| `block_size` | 32 | Max sequence / context length |
| `learning_rate` | 1e-3 | Adam optimizer LR |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 2 | Attention heads |
| `n_layer` | 4 | Transformer layers |
| `epochs_CLS` | 15 | Classification training epochs |
| `max_iters` | 500 | LM training iterations |
| `eval_interval` | 100 | LM perplexity eval frequency |

---

## Dataset Format

**Classification (`train_CLS.tsv`, `test_CLS.tsv`):** Tab-separated, one sample per line.
```
0	We must invest in our schools and our teachers.
1	We will work to keep America safe and free.
2	The strength of our nation lies in our people.
```
Labels: `0` = Obama, `1` = W. Bush, `2` = H. Bush

**Language Modeling (`train_LM.txt`, `test_LM_*.txt`):** Plain text files used for next-token prediction. Each sample is a sliding window of `block_size` tokens.

---

## References

- Press et al. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022.
- Beltagy et al. (2020). *Longformer: The Long-Document Transformer.* arXiv:2004.05150.
- He et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* ICLR 2021.
- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS 2017.