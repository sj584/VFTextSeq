import argparse
import os

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

from .preprocess import fasta_to_df

# ESM2
def esm_extract(df, model="esm2_t33_650M_UR50D", layer=33, batch_size=1, device="cuda:5"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model)
    batch_converter = alphabet.get_batch_converter()    
    model = model.to(device)
    model.eval()
    data = list(zip(df.id.values, df.sequence.values))
    sequence_representations = []
    for start_idx in tqdm(range(0, len(data), batch_size), total=len(data) // batch_size):
        batch_data = data[start_idx:start_idx + batch_size]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        token_representations = results["representations"][layer]

        for i, tokens_len in enumerate(batch_lens):
            embeddings = token_representations[i, 1: tokens_len - 1]
            emb_sum = embeddings.mean(0)
            sequence_representations.append(emb_sum.cpu().numpy().tolist())
    X = np.array(sequence_representations)
    return X


def main():
    parser = argparse.ArgumentParser(description='Generate ESM2 sequence-level embeddings from FASTA')
    parser.add_argument('--fasta_path', '-i', type=str, required=True, help='Input FASTA file path')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Output directory for pooled embeddings')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = os.path.join(args.output_dir, f"esm_embedding.npy")

    # Skip if embedding already exists
    if os.path.exists(output_file):
        print(f"Sequence-level embedding for {args.fasta_path} already exists. Skipping.")
        pass

    else:
        df = fasta_to_df(args.fasta_path)
        seq_emb = esm_extract(df)
        np.save(output_file, seq_emb)

if __name__ == "__main__":

    main()
