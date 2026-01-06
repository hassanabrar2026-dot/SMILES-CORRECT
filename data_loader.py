import re

def parse_uncharacterized_file(path):
    """
    Parses the uncharacterized.txt file and returns
    a list of (input_smiles, target_smiles) pairs.
    Here: input = B3LYP SMILES, target = GDB17 SMILES.
    """
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_started = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("#") and "Index" in line and "GDB17" in line:
            data_started = True
            continue

        if not data_started:
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 4:
            continue

        index = parts[0]
        gdb17_smiles = parts[1]
        b3lyp_smiles = parts[2]
        corina_smiles = parts[3]

        input_smiles = b3lyp_smiles
        target_smiles = gdb17_smiles

        pairs.append((input_smiles, target_smiles))

    return pairs


if __name__ == "__main__":
    path = "uncharacterized.txt"
    pairs = parse_uncharacterized_file(path)
    print(f"Loaded {len(pairs)} pairs.")
    for i in range(min(5, len(pairs))):
        print(pairs[i])
