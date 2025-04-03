from collections import Counter
import re

class Tokenizer:
    def __init__(self, min_freq=1, lower=True):
        self.min_freq = min_freq
        self.lower = lower
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        self.vocab_built = False

    def _tokenize(self, text):
        if self.lower:
            text = text.lower()
        return re.findall(r"\b\w+\b", text)  # regex = split on word boundaries

    def build_vocab(self, captions):
        counter = Counter()
        for caption in captions:
            tokens = self._tokenize(caption)
            counter.update(tokens)

        # Start index after special tokens
        self.word2idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        idx = len(self.word2idx)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_built = True

    def text_to_sequence(self, text):
        if not self.vocab_built:
            raise ValueError("Vocab not built. Call build_vocab() first.")
        
        tokens = self._tokenize(text)
        seq = [self.word2idx.get('<start>')]

        for tok in tokens:
            idx = self.word2idx.get(tok, self.word2idx['<unk>'])
            seq.append(idx)

        seq.append(self.word2idx.get('<end>'))
        return seq

    def pad_sequence(self, seq, max_len):
        pad_idx = self.word2idx['<pad>']
        if len(seq) < max_len:
            seq += [pad_idx] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

    def __len__(self):
        return len(self.word2idx)