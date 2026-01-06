import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd

from dataset import SMILESDataset
from model_transformer import TransformerSeq2Seq

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESULTS_CSV = "training_results.csv"

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return inputs, targets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = SMILESDataset("uncharacterized.txt")
    vocab_size = len(full_dataset.char_to_idx)
    pad_idx = full_dataset.pad_idx

    total_len = len(full_dataset)
    val_len = max(1, int(0.1 * total_len))
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = TransformerSeq2Seq(vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", ncols=100)
        for inp, tgt in pbar:
            inp = inp.to(device)
            tgt = tgt.to(device)

            decoder_input = tgt[:, :-1]
            target_output = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(inp, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                decoder_input = tgt[:, :-1]
                target_output = tgt[:, 1:]

                logits = model(inp, decoder_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time_sec": elapsed
        })

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "char_to_idx": full_dataset.char_to_idx,
        }, ckpt_path)

    df = pd.DataFrame(history)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved training history to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
