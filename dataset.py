import torch
from torch.utils.data import Dataset
from data_loader import parse_uncharacterized_file

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]

class SMILESDataset(Dataset):
    def __init__(self, path, char_to_idx=None):
        self.pairs = parse_uncharacterized_file(path)

        # Build vocabulary
        if char_to_idx is None:
            chars = set()
            for inp, tgt in self.pairs:
                chars.update(list(inp))
                chars.update(list(tgt))
            chars = sorted(list(chars))

            # Reserve indices for special tokens
            self.char_to_idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
            offset = len(SPECIAL_TOKENS)
            for i, c in enumerate(chars):
                self.char_to_idx[c] = i + offset
        else:
            self.char_to_idx = char_to_idx

        # Reverse mapping
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        # Special token indices
        self.pad_idx = self.char_to_idx["<PAD>"]
        self.sos_idx = self.char_to_idx["<SOS>"]
        self.eos_idx = self.char_to_idx["<EOS>"]

    def encode(self, smiles, add_sos_eos=True):
        tokens = [self.char_to_idx[c] for c in smiles]
        if add_sos_eos:
            tokens = [self.sos_idx] + tokens + [self.eos_idx]
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, indices):
        chars = []
        for idx in indices:
            idx = int(idx)
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            chars.append(self.idx_to_char.get(idx, ""))
        return "".join(chars)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp, tgt = self.pairs[idx]
        return self.encode(inp), self.encode(tgt)
