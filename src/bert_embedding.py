import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import os
import argparse

def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    embedding = last_hidden_states.mean(dim=1).squeeze()
    return embedding.cpu()

def embed_and_save(data_type, df, output_dir, tokenizer, model, device):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {output_dir}"):
        if data_type == "lca":
            desc = str(row.get('name', '')).strip()
            rank = str(row.get('rank', '')).strip()
            combined_text = f"{rank}. {desc}" if desc or rank else 'NA'
        elif data_type == "interproscan":
            desc = str(row.get('desc_nodup', '')).strip()
            combined_text = desc if desc else 'NA'
        else:
            ValueError("Data type must be either 'interproscan' or 'lca'")
        
        embedding = get_embedding(combined_text, tokenizer, model, device)
        file_path = os.path.join(output_dir, f"{row['id']}.pt")
        torch.save(embedding, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BERT embeddings")
    parser.add_argument("--input", "-i", required=True, help="Path to input InterProScan or LCA .csv file")
    parser.add_argument("--type", "-t", default="lca", required=True, help="'interproscan' or 'lca'")
    parser.add_argument("--output", "-o", default="./", help="Path to output .pt embeddings")
    args = parser.parse_args()
        
    df = pd.read_csv(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    model.eval()
    
    print("Input CSV:", args.input)
    print("Output dir:", args.output)
    os.makedirs(args.output, exist_ok=True)
    embed_and_save(args.type, df, args.output_dir, tokenizer, model, device)