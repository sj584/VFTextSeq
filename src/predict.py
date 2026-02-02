#!/usr/bin/env python3
"""
predict.py - Predict virulence factors using best_binary_xgb_model.pkl
Usage: python predict.py --input_csv example/example.csv --output result.csv --esm_emb example/esm_emb --interproscan_emb example/interproscan_bert_emb --tax_emb example/tax_bert_emb
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import fasta_to_df


def load_embeddings(emb_dirs):
    emb_list = []

    for path in Path(emb_dirs).glob("*.npy"):
        emb_list.append(np.load(path))

    if len(emb_list) == 0:
        assert ValueError("No embedding found! Please generate your embedding first.")
    
    emb = np.concatenate(emb_list, axis=1)

    return emb

def main():
    parser = argparse.ArgumentParser(description='Predict using trained XGBoost model')
    parser.add_argument('--input_dir', '-i', type=str, required=True, help='Preprocessed CSV file path')
    parser.add_argument("--embedding_dir", "-e", required=True, help="Path to sequence and text embedding .npy files")
    parser.add_argument('--model_path', default='best_binary_xgb_model.pkl',
                        help='Path to XGBoost model pickle file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction (default: 0.5) [0,1]')
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: same directory as input file)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"{args.model_path} not found.")

    print("Loading XGBoost model...")
    loaded_model = joblib.load(args.model_path)

    print("Loading embeddings...")
    print(f"Processing {args.input_dir}...")
    input_path = Path(args.input_dir)
    if input_path.suffix.lower() in {".fasta", ".fa", ".faa"}:
        df = fasta_to_df(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    file_name = input_path.stem
    X = load_embeddings(args.embedding_dir)

    print("Making predictions...")
    probabilities = loaded_model.predict_proba(X)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    df["prob"] = probabilities
    df["pred"] = predictions

    input_path = Path(args.input_dir)
    if args.output_dir is None:
        args.output_dir = os.path.join(str(input_path.parent), "results")
        
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving results to: {args.output_dir}")
    output_dir = os.path.join(args.output_dir, f"{file_name}_results.csv")
    df.to_csv(output_dir, index=False)
    print(f"Prediction complete! Shape: {df.shape} (threshold={args.threshold})")
    print(df.head())

if __name__ == '__main__':
    main()
