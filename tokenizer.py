from nltk.tokenize import word_tokenize
import os


class SimpleTokenizer:
    """
    A simple tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self, text):
        """Initialize the tokenizer with the initial text to build vocabulary."""
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.build_vocab(text)

    def build_vocab(self, text):
        """Build vocabulary from the given text."""
        tokens = word_tokenize(text)
        self.vocab = set(tokens)
        # Reserve 0: <pad>, 1: <unk>, 2: <cls>
        self.vocab_size = len(self.vocab) + 3
        self.stoi = {word: i for i, word in enumerate(self.vocab, start=3)}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.stoi['<cls>'] = 2
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text, add_cls=False):
        """Encode the text into a list of indices. Optionally prepend <cls> token."""
        tokens = word_tokenize(text)
        indices = [self.stoi.get(word, self.stoi['<unk>']) for word in tokens]
        if add_cls:
            indices = [self.stoi['<cls>']] + indices
        return indices

    def decode(self, indices, skip_special=True):
        """Decode the list of indices back into text. Optionally skip special tokens."""
        words = []
        for index in indices:
            word = self.itos.get(index, '<unk>')
            if skip_special and word in {'<pad>', '<unk>', '<cls>'}:
                continue
            words.append(word)
        return ' '.join(words)
    
