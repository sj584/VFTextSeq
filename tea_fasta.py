#!/usr/bin/env python3
"""
TEA tokenization script for FASTA sequences with entropy calculation.
Usage: python tea_fasta.py price.fasta price_tea.fasta
"""

import argparse
import torch
import re
from pathlib import Path
from Bio import SeqIO
from tea.model import Tea
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig

def load_models(device):
    """Load TEA and ESM2 models - ensure all on same device."""
    print("Loading TEA model...")
    tea = Tea.from_pretrained("PickyBinders/tea")
    tea.to(device)  # Move TEA to device AFTER loading
    tea.eval()
    
    print("Loading ESM2 model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    esm2 = AutoModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D",
        torch_dtype="auto",
        quantization_config=bnb_config,
        add_pooling_layer=False,
    ).to(device)
    esm2.eval()
    
    return tea, esm2, tokenizer

def preprocess_sequences(sequences):
    """Preprocess sequences for ESM2 tokenization."""
    return [" ".join(list(re.sub(r"[UZOBJ]", "X", sequence))) for sequence in sequences]

def process_batch(sequences, tea, esm2, tokenizer, device, batch_size=16):
    """Process sequences in batches to avoid OOM."""
    results = {'sequences': [], 'avg_entropy': []}
    
    processed_seqs = preprocess_sequences(sequences)
    
    for i in range(0, len(processed_seqs), batch_size):
        batch_seqs = processed_seqs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(processed_seqs)-1)//batch_size + 1}")
        
        # Tokenize batch
        ids = tokenizer(batch_seqs, add_special_tokens=True, padding="longest", return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        # Clear cache before inference
        torch.cuda.empty_cache()
        
        # Get embeddings and TEA sequences
        with torch.no_grad():
            x = esm2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            batch_results = tea.to_sequences(
                embeddings=x, 
                input_ids=input_ids, 
                return_avg_entropy=True, 
                return_logits=False, 
                return_residue_entropy=False
            )
        
        results['sequences'].extend(batch_results['sequences'])
        # FIXED: avg_entropy is already a list, no .tolist() needed
        results['avg_entropy'].extend(batch_results['avg_entropy'])
    
    return results

def process_fasta(input_fasta, output_fasta, batch_size=16):
    """Process FASTA file through TEA pipeline with batching."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Load models
    tea, esm2, tokenizer = load_models(device)
    
    # Read sequences
    records = list(SeqIO.parse(input_fasta, "fasta"))
    sequences = [str(record.seq) for record in records]
    print(f"Loaded {len(sequences)} sequences")
    
    # Process in batches
    results = process_batch(sequences, tea, esm2, tokenizer, device, batch_size)
    
    # Write output FASTA with entropy in headers
    with open(output_fasta, "w") as out_handle:
        for i, (record, avg_entropy) in enumerate(zip(records, results['avg_entropy'])):
            header = f">{record.id}|avg_entropy={avg_entropy:.6f}"
            if hasattr(record, 'description') and record.description and record.id != record.description.split()[0]:
                header += f" {record.description.replace(record.id, '').strip()}"
            seq = results['sequences'][i]
            
            out_handle.write(f"{header}\n{seq}\n")
    
    print(f"Saved {len(records)} sequences to {output_fasta}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA sequences to TEA tokens with entropy")
    parser.add_argument("input_fasta", help="Input FASTA file")
    parser.add_argument("output_fasta", help="Output FASTA file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing (default: 16)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_fasta)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    process_fasta(args.input_fasta, args.output_fasta, args.batch_size)

