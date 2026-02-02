import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ESM2
def seq_extract(df, model_name="esm2_t33_650M_UR50D", layer=33, batch_size=1, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    data = list(zip(df.id.values, df.sequence.values))
    sequence_representations = []
    for start_idx in tqdm(range(0, len(data), batch_size), total=len(data) // batch_size, desc=f"Extracting sequence embedding"):
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
    return np.array(sequence_representations)


def txt_extract(df, model_name="google-bert/bert-base-uncased", device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    text_representations = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting text embedding"):
        desc = str(row.get('desc_nodup', '')).strip()
        name = str(row.get('name', '')).strip()
        rank = str(row.get('rank', '')).strip()
        combined_text = f"{desc}|{name}|{rank}"
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states.mean(dim=1).squeeze().cpu()
        text_representations.append(embedding.numpy().tolist())
    return np.array(text_representations)

def main():
    parser = argparse.ArgumentParser(description='Generate pretrained model embeddings from FASTA')
    parser.add_argument('--input_dir', '-i', type=str, required=True, help='Preprocessed CSV file path')
    parser.add_argument('--seq_model', '-sm', default="esm2-650m", type=str, help='Name of pretrained protein language model')
    parser.add_argument('--text_model', '-tm', default="bert", type=str, help='Name of pretrained language model')
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: same directory as input file)"
    )
    
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if args.output_dir is None:
        args.output_dir = os.path.join(str(input_path.parent), "embedding")
        
    os.makedirs(args.output_dir, exist_ok=True)
    file_name = Path(args.input_dir).stem
    df = pd.read_csv(args.input_dir)

    seq_out = os.path.join(args.output_dir, f"{file_name}_seq_embedding.npy")
    if os.path.exists(seq_out):
        print(f"Sequence embedding for {args.input_dir} already exists. Skipping.")
    else:
        if args.seq_model == "esm2-650m":
            seq_emb = seq_extract(df)
        elif args.seq_model == "esm2-8m":
            seq_emb = seq_extract(df, model_name="esm2_t6_8M_UR50D", layer=6)
        elif args.seq_model == "esm2-35m":
            seq_emb = seq_extract(df, model_name="esm2_t12_35M_UR50D", layer=12)
        elif args.seq_model == "esm2-150m":
            seq_emb = seq_extract(df, model_name="esm2_t30_150M_UR50D", layer=30)
        else:
            print("Model name unrecognized\n")
            print("Processing sequence with default setting (ESM2-650M)...")
            seq_emb = seq_extract(df)
        
        np.save(seq_out, seq_emb)

    txt_out = os.path.join(args.output_dir, f"{file_name}_txt_embedding.npy")
    if os.path.exists(txt_out):
        print(f"Text embedding for {args.input_dir} already exists. Skipping.")
    else:
        if args.text_model == "bert":
            txt_emb = txt_extract(df)
        elif args.text_model == "biobert":
            txt_emb = txt_extract(df, model_name="dmis-lab/biobert-v1.1")
        elif args.text_model == "mistral":
            txt_emb = txt_extract(df, model_name="mistralai/Mistral-7B-v0.1")
        elif args.text_model == "biomistral":
            txt_emb = txt_extract(df, model_name="BioMistral/BioMistral-7B")
        else:
            print("Model name unrecognized\n")
            print("Processing bio-text with default setting (BERT)...")
            txt_emb = txt_extract(df)
        
        np.save(txt_out, txt_emb)

    print(f"Saved sequence and text embeddings to {args.output_dir}")

if __name__ == "__main__":
    main()