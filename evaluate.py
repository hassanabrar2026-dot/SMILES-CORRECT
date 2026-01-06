import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
from dataset import SMILESDataset
from model_transformer import TransformerSeq2Seq
from predict import greedy_decode
from tqdm import tqdm

def tanimoto(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SMILESDataset("uncharacterized.txt")
    vocab_size = len(dataset.char_to_idx)

    model = TransformerSeq2Seq(vocab_size=vocab_size, pad_idx=dataset.pad_idx).to(device)
    ckpt = torch.load("checkpoints/model_epoch_20.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    total = 0
    valid = 0
    exact_match = 0
    avg_lev = 0
    avg_tanimoto = 0
    tanimoto_count = 0

    for inp, tgt in tqdm(dataset.pairs, desc="Evaluating"):
        encoded = dataset.encode(inp).unsqueeze(0).to(device)
        out_tokens = greedy_decode(model, encoded, dataset.sos_idx, dataset.eos_idx)
        pred = dataset.decode(out_tokens.tolist())

        total += 1

        # Validity
        if Chem.MolFromSmiles(pred) is not None:
            valid += 1

        # Exact match
        if pred == tgt:
            exact_match += 1

        # Levenshtein
        import Levenshtein
        avg_lev += Levenshtein.distance(pred, tgt)

        # Tanimoto
        sim = tanimoto(pred, tgt)
        if sim is not None:
            avg_tanimoto += sim
            tanimoto_count += 1

    print("\n=== Evaluation Results ===")
    print("Total samples:", total)
    print("Validity:", valid / total)
    print("Exact match:", exact_match / total)
    print("Avg Levenshtein distance:", avg_lev / total)
    print("Avg Tanimoto similarity:", avg_tanimoto / tanimoto_count)

if __name__ == "__main__":
    main()
