import torch
from model_transformer import TransformerSeq2Seq
from dataset import SMILESDataset
import argparse

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt

def greedy_decode(model, src, sos_idx, eos_idx, max_len=200):
    model.eval()
    device = src.device

    tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model(src, tgt)
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == eos_idx:
            break

    return tgt.squeeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temp_dataset = SMILESDataset("uncharacterized.txt")
    char_to_idx = temp_dataset.char_to_idx
    idx_to_char = temp_dataset.idx_to_char
    pad_idx = temp_dataset.pad_idx
    sos_idx = temp_dataset.sos_idx
    eos_idx = temp_dataset.eos_idx

    vocab_size = len(char_to_idx)

    model = TransformerSeq2Seq(vocab_size=vocab_size, pad_idx=pad_idx).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state_dict"])

    encoded = temp_dataset.encode(args.input).unsqueeze(0).to(device)

    output_tokens = greedy_decode(model, encoded, sos_idx, eos_idx)
    output_smiles = temp_dataset.decode(output_tokens.tolist())

    print("Input SMILES:     ", args.input)
    print("Predicted SMILES: ", output_smiles)

if __name__ == "__main__":
    main()
