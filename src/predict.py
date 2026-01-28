#!/usr/bin/env python3
"""
predict.py - Predict virulence factors using best_binary_xgb_model.pkl
Usage: python predict.py --input_csv example/example.csv --output result.csv --esm_emb example/esm_emb --interproscan_emb example/interproscan_bert_emb --tax_emb example/tax_bert_emb
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm


def load_multiple_embeddings(fasta_path, emb_dirs):
    """
    Load and concatenate embeddings for FASTA record IDs.
    Args:
        fasta_path: Path to FASTA file
        emb_dirs: list of 3 paths to embedding directories
    Returns:
        features: np.ndarray of concatenated embeddings
        ids: list of corresponding FASTA record IDs
    """
    # Parse FASTA to get IDs
    ids = [record.id for record in SeqIO.parse(fasta_path, "fasta")]

    features = []
    for id_ in tqdm(ids, desc="Loading embeddings"):
        emb_list = []
        for emb_dir in emb_dirs:
            emb_path = os.path.join(emb_dir, f'{id_}.pt')
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"Embedding file not found: {emb_path}")
            emb = torch.load(emb_path)
            emb_np = emb.numpy()

            if emb_np.ndim == 2:
                emb_np = emb_np.mean(axis=0)
            elif emb_np.ndim != 1:
                raise ValueError(f'Unexpected embedding shape: {emb_np.shape} for id {id_} in {emb_dir}')

            emb_list.append(emb_np)

        concatenated_emb = np.concatenate(emb_list)
        features.append(concatenated_emb)

    features = np.stack(features)
    return features, ids

def main():
    parser = argparse.ArgumentParser(description='Predict using trained XGBoost model')
    parser.add_argument('--input_fasta', required=True,
                        help='Path to input FASTA')
    parser.add_argument('--output', required=True,
                        help='Output CSV path for predictions')
    parser.add_argument('--model_path', default='best_binary_xgb_model.pkl',
                        help='Path to XGBoost model pickle file')
    parser.add_argument('--esm_emb', required=True,
                        help='Directory with ESM embeddings (*.pt)')
    parser.add_argument('--interproscan_emb', required=True,
                        help='Directory with InterProScan BERT embeddings (*.pt)')
    parser.add_argument('--tax_emb', required=True,
                        help='Directory with taxonomy BERT embeddings (*.pt)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction (default: 0.5) [0,1]')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"{args.model_path} not found")

    print("Loading XGBoost model...")
    loaded_model = joblib.load(args.model_path)

    emb_dirs = [args.esm_emb, args.interproscan_emb, args.tax_emb]
    print("Loading embeddings...")
    X, ids = load_multiple_embeddings(args.input_fasta, emb_dirs)

    print("Making predictions...")
    probabilities = loaded_model.predict_proba(X)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    result_df = pd.DataFrame({
        'id': ids,
        'prob': probabilities,
        'pred': predictions,
    })

    print(f"Saving results to: {args.output}")
    result_df.to_csv(args.output, index=False)
    print(f"Prediction complete! Shape: {result_df.shape} (threshold={args.threshold})")
    print(result_df.head())

if __name__ == '__main__':
    main()
