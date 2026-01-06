#This project implements a Transformer‑based SMILES error‑correction model trained on corrupted molecular strings.
The workflow includes:

Dataset preparation

Model training

Prediction generation

Statistical evaluation

CSV‑based analysis

High‑resolution (1200 dpi) visualizations

All results are saved as CSV or PNG files for easy downstream analysis
#SMILES/
│
├── dataset.py
├── model_transformer.py
├── train.py
│
├── generate_all_csvs.py
├── generate_confusion_matrix_csv.py
├── plot_confusion_matrix_csv.py
├── plot_confusion_matrix_heatmap.py
│
├── scatter_data.csv
├── distribution_data.csv
├── best_predictions.csv
├── worst_predictions.csv
├── confusion_matrix.csv
│
├── confusion_matrix_1200dpi.png
├── confusion_matrix_heatmap_1200dpi.png
│
└── analysis_results.xlsx
#The model is a Transformer Seq2Seq architecture trained to convert corrupted SMILES strings into valid, corrected SMILES.

Key features:

Character‑level tokenization

Encoder–decoder Transformer

Greedy decoding for inference

RDKit‑based molecular validation

Tanimoto similarity scoring

Levenshtein distance evaluation