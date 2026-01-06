from rdkit import Chem
from rdkit.Chem import Descriptors

def validate(smiles):
    print("Input:", smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("❌ Invalid SMILES")
        return

    print("✔ Valid SMILES")

    try:
        Chem.SanitizeMol(mol)
        print("✔ Sanitized successfully")
    except Exception as e:
        print("⚠ Sanitization warning:", e)

    print("Formula:", Chem.rdMolDescriptors.CalcMolFormula(mol))
    print("MW:", Descriptors.MolWt(mol))
    print("Num atoms:", mol.GetNumAtoms())
    print("Num bonds:", mol.GetNumBonds())

if __name__ == "__main__":
    validate("CC[NH2+]CCCCC([O-])CO")
