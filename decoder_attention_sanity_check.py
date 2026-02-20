from utilities import Utilities
from transformer import TransformerDecoder
from tokenizer import SimpleTokenizer
import torch

if __name__ == "__main__":
    # Example sentences for decoder attention visualization
    sentences = [
        "That is in Israel's interest, Palestine's interest, America's interest, and the world's interest.",
        "They want to work, and they can work, and this is a tremendous pool of people."
    ]
    block_size = 32

    # Build vocab from sentences
    texts = [sentences[0] + " " + sentences[1]]
    tokenizer = SimpleTokenizer(' '.join(texts))
    decoder = TransformerDecoder(tokenizer.vocab_size, max_seq_len=block_size)

    class DecoderWithAttn(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
        def forward(self, x):
            # Only return self-attention maps
            logits, attn_maps = self.decoder(x, return_attn=True)
            # attn_maps: list of (batch, num_heads, seq, seq)
            flat_maps = [m.mean(1, keepdim=True) for m in attn_maps] if attn_maps else []
            return logits, flat_maps

    decoder_with_attn = DecoderWithAttn(decoder)
    utils = Utilities(tokenizer, decoder_with_attn)

    for sent in sentences:
        print(f"Sanity checking decoder attention for: {sent}")
        utils.sanity_check(sent, block_size)
