import torch
from esm import pretrained
from Bio import SeqIO
import os
import argparse

parser = argparse.ArgumentParser(description='Generate ESM2 sequence-level embeddings from FASTA')
parser.add_argument('--fasta_path', type=str, required=True, help='Input FASTA file path')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for pooled embeddings')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()

batch_converter = alphabet.get_batch_converter()
MAX_LEN = 1022

for record in SeqIO.parse(args.fasta_path, "fasta"):
    header = record.id
    output_file = os.path.join(args.output_dir, f"{header}.pt")

    # Skip if embedding already exists
    if os.path.exists(output_file):
        print(f"Sequence-level embedding for {header} already exists. Skipping.")
        continue

    sequence = str(record.seq)

    if len(sequence) == 0:
        print(f"Warning: sequence {header} empty. Skipping.")
        continue

    if len(sequence) <= MAX_LEN:
        # Single run
        data = [(header, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            residue_emb = token_representations[0, 1:len(sequence)+1, :].cpu()
    else:
        # Over MAX_LEN: split, embed twice, then concat
        seq1 = sequence[:MAX_LEN]
        seq2 = sequence[MAX_LEN:]
        # First chunk
        data1 = [(header, seq1)]
        batch_labels1, batch_strs1, batch_tokens1 = batch_converter(data1)
        batch_tokens1 = batch_tokens1.to(device)
        with torch.no_grad():
            results1 = model(batch_tokens1, repr_layers=[33], return_contacts=False)
            emb1 = results1["representations"][33][0, 1:len(seq1)+1, :].cpu()
        # Second chunk
        data2 = [(header, seq2)]
        batch_labels2, batch_strs2, batch_tokens2 = batch_converter(data2)
        batch_tokens2 = batch_tokens2.to(device)
        with torch.no_grad():
            results2 = model(batch_tokens2, repr_layers=[33], return_contacts=False)
            emb2 = results2["representations"][33][0, 1:len(seq2)+1, :].cpu()
        residue_emb = torch.cat([emb1, emb2], dim=0)

    # Average pooling to get sequence-level embedding
    seq_emb = residue_emb.mean(dim=0)
    print(header, seq_emb.shape)

    # Save sequence-level embedding
    torch.save(seq_emb, output_file)
