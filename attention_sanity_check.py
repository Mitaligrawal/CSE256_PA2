from utilities import Utilities
from transformer import TransformerEncoder
from tokenizer import SimpleTokenizer
import torch

if __name__ == "__main__":
    # Example sentences for attention visualization
    sentences = [
        "That is in Israel's interest, Palestine's interest, America's interest, and the world's interest.",
        "They want to work, and they can work, and this is a tremendous pool of people."
    ]
    block_size = 32

    # Load tokenizer and encoder as in main.py
    texts = [sentences[0] + " " + sentences[1]]  # Just to build vocab
    tokenizer = SimpleTokenizer(' '.join(texts))
    encoder = TransformerEncoder(tokenizer.vocab_size, max_seq_len=block_size)

    # Patch encoder to return attention maps for utilities.py compatibility
    class EncoderWithAttn(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, x):
            # Get hooks for attention weights
            attn_maps = []
            def hook_fn(module, input, output):
                if hasattr(module, 'attn_output_weights'):
                    attn_maps.append(module.attn_output_weights.detach().cpu())
            handles = []
            for layer in self.encoder.transformer_encoder.layers:
                handles.append(layer.self_attn.register_forward_hook(hook_fn))
            out = self.encoder(x)
            for h in handles:
                h.remove()
            # attn_maps: list of (batch, num_heads, seq, seq)
            # For utilities.py, flatten heads: (batch, seq, seq) per map
            flat_maps = [m.mean(1, keepdim=True) for m in attn_maps]  # mean over heads
            return out, flat_maps

    encoder_with_attn = EncoderWithAttn(encoder)
    utils = Utilities(tokenizer, encoder_with_attn)

    for sent in sentences:
        print(f"Sanity checking attention for: {sent}")
        utils.sanity_check(sent, block_size)
