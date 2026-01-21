#!/usr/bin/env python3
"""
predict.py - Predict virulence factors using best_binary_xgb_model.pkl
Usage: python predict.py --input_csv example/example.csv --output result.csv --esm_emb example/esm_emb --interproscan_emb example/interproscan_bert_emb --tax_emb example/tax_bert_emb
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

def load_multiple_embeddings(df, emb_dirs):
    """
    Load and concatenate embeddings from multiple directories for the same ids in df.
    Args:
        df: DataFrame with columns ['id', 'label']
        emb_dirs: list of 3 paths to embedding directories
    Returns:
        features: np.ndarray of concatenated embeddings from all three sources
        labels: np.ndarray of labels
    """
    features = []
    labels = []

    for idx in tqdm(range(len(df)), desc="Loading embeddings"):
        id_ = df.iloc[idx]['id']
        label = df.iloc[idx]['label']

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
        labels.append(label)

    features = np.stack(features)
    labels = np.array(labels)
    return features, labels

def main():
    parser = argparse.ArgumentParser(description='Predict using trained XGBoost model')
    parser.add_argument('--input_csv', required=True, help='Path to input CSV with id and label columns')
    parser.add_argument('--output', required=True, help='Output CSV path for predictions')
    parser.add_argument('--esm_emb', required=True, help='Directory with ESM embeddings (*.pt)')
    parser.add_argument('--interproscan_emb', required=True, help='Directory with InterProScan BERT embeddings (*.pt)')
    parser.add_argument('--tax_emb', required=True, help='Directory with taxonomy BERT embeddings (*.pt)')
    args = parser.parse_args()

    # Verify model exists
    if not os.path.exists('best_binary_xgb_model.pkl'):
        raise FileNotFoundError("best_binary_xgb_model.pkl not found in current directory")

    # Load model
    print("Loading XGBoost model...")
    loaded_model = joblib.load('best_binary_xgb_model.pkl')

    # Load input CSV
    print(f"Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if not all(col in df.columns for col in ['id', 'label']):
        raise ValueError("Input CSV must have 'id' and 'label' columns")

    # Load embeddings
    emb_dirs = [args.esm_emb, args.interproscan_emb, args.tax_emb]
    print("Loading embeddings...")
    X, y_labels = load_multiple_embeddings(df, emb_dirs)

    # Predict
    print("Making predictions...")
    probabilities = loaded_model.predict_proba(X)[:, 1]
    predictions = loaded_model.predict(X)

    # Prepare result dataframe
    result_df = pd.DataFrame({
        'id': df['id'],
        'prob': probabilities,
        'pred': predictions,
        'label': y_labels
    })

    # Save results
    print(f"Saving results to: {args.output}")
    result_df.to_csv(args.output, index=False)
    print(f"Prediction complete! Shape: {result_df.shape}")
    print(result_df.head())

if __name__ == '__main__':
    main()

