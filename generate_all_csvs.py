import pandas as pd
import torch
import Levenshtein

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from dataset import SMILESDataset
from model_transformer import TransformerSeq2Seq


# -----------------------------
# Greedy decoding
# -----------------------------
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


# -----------------------------
# Tanimoto similarity
# -----------------------------
def tanimoto(a, b):
    m1, m2 = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if m1 is None or m2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# -----------------------------
# Main script
# -----------------------------
def main():
    print("Loading dataset and model...")

    dataset = SMILESDataset("uncharacterized.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerSeq2Seq(
        len(dataset.char_to_idx),
        pad_idx=dataset.pad_idx
    ).to(device)

    ckpt = torch.load("checkpoints/model_epoch_20.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # CSV containers
    scatter_rows = []
    distribution_rows = []
    tanimoto_rows = []

    print("Generating predictions...")

    for inp, tgt in dataset.pairs:
        encoded = dataset.encode(inp).unsqueeze(0).to(device)
        out_tokens = greedy_decode(model, encoded, dataset.sos_idx, dataset.eos_idx)
        pred = dataset.decode(out_tokens.tolist()).strip()

        if not pred:
            continue

        # Scatter data (Levenshtein)
        lev = Levenshtein.distance(pred, tgt)
        scatter_rows.append({
            "input": inp,
            "target": tgt,
            "prediction": pred,
            "target_length": len(tgt),
            "levenshtein": lev
        })

        # Distribution data
        mol = Chem.MolFromSmiles(pred)
        valid = bool(mol)
        sim = tanimoto(pred, tgt)

        distribution_rows.append({
            "input": inp,
            "target": tgt,
            "prediction": pred,
            "pred_length": len(pred),
            "valid": valid,
            "tanimoto": sim
        })

        if sim is not None:
            tanimoto_rows.append((sim, inp, tgt, pred))

    # -----------------------------
    # Save scatter CSV
    # -----------------------------
    pd.DataFrame(scatter_rows).to_csv("scatter_data.csv", index=False)
    print("Saved scatter_data.csv")

    # -----------------------------
    # Save distribution CSV
    # -----------------------------
    pd.DataFrame(distribution_rows).to_csv("distribution_data.csv", index=False)
    print("Saved distribution_data.csv")

    # -----------------------------
    # Save best/worst predictions
    # -----------------------------
    if tanimoto_rows:
        tanimoto_rows.sort(key=lambda x: x[0], reverse=True)

        best = tanimoto_rows[:12]
        worst = tanimoto_rows[-12:]

        pd.DataFrame(best, columns=["tanimoto", "input", "target", "prediction"]).to_csv(
            "best_predictions.csv", index=False
        )

        pd.DataFrame(worst, columns=["tanimoto", "input", "target", "prediction"]).to_csv(
            "worst_predictions.csv", index=False
        )

        print("Saved best_predictions.csv and worst_predictions.csv")
    else:
        print("No valid Tanimoto similarities found — skipping best/worst CSVs.")


if __name__ == "__main__":
    main()
