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

def embed_and_save(df, output_dir, tokenizer, model, device):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {output_dir}"):
        desc = str(row.get('desc_nodup', '')).strip()
        combined_text = desc if desc else 'NA'
        embedding = get_embedding(combined_text, tokenizer, model, device)
        file_path = os.path.join(output_dir, f"{row['id']}.pt")
        torch.save(embedding, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BERT embeddings from InterProScan CSV")
    parser.add_argument("--input_csv", help="Path to input CSV file (e.g., example_Interproscan_semantic_removal.csv)")
    parser.add_argument("--output_dir", help="Output directory for .pt embeddings")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df_total = pd.read_csv(args.input_csv)
    
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    model.eval()
    
    print("Input CSV:", args.input_csv)
    print("Output dir:", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    embed_and_save(df_total, args.output_dir, tokenizer, model, device)
