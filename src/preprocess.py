import argparse

import pandas as pd
from Bio import SeqIO
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def fasta_to_df(path, id_field="id"):
    rows = []
    for rec in SeqIO.parse(path, "fasta"):
        seq_id = rec.id if id_field == "id" else rec.description
        rows.append((seq_id, str(rec.seq)))
    return pd.DataFrame(rows, columns=["id", "sequence"])

def parse_interproscan_file(tsv_file):
    df = pd.read_csv(tsv_file, sep="\t", header=None, comment="#")
    
    # Set column names according to InterProScan standard [web:7]
    df.columns = [
        "id",      # 0
        "Sequence MD5 Digest",    # 1
        "Sequence Length",        # 2
        "Analysis",               # 3
        "Signature Accession",    # 4
        "Signature Description",  # 5
        "Start",                  # 6
        "Stop",                   # 7
        "Score",                  # 8
        "Status",                 # 9
        "Date",                   # 10
        "InterPro Accession",     # 11
        "InterPro Description",   # 12
        "GO Annotations",         # 13
        "Pathway Annotations"     # 14
    ]
    return df

def parse_mmseqs_lca(filepath):
    records = []
    last_id = None
    with open(filepath, encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            if "\t" not in line:
                # ID-only line, e.g. VFG029903
                last_id = line
            else:
                parts = line.split("\t")
                # If first part is an ID (not empty), use it
                if parts[0] and not parts[0].isspace():
                    rec_id = parts[0]
                    taxid, rank, name = parts[1:]
                else:
                    # First part empty: use last saved ID
                    rec_id = last_id
                    taxid, rank, name = parts[1:]
                records.append({
                    "id": rec_id,
                    "taxid": taxid,
                    "rank": rank,
                    "name": name,
                })
                last_id = None
    return pd.DataFrame(records)


def build_interpro_description(df_inter, id_col, desc_cols):
    description_map = {}
    for entry, group in df_inter.groupby(id_col):
        descs = set()
        for col in desc_cols:
            if col in group:
                for val in group[col].dropna():
                    if val != "-":
                        descs.add(str(val))
        description_map[entry] = "|".join(sorted(descs))
    return description_map

def remove_semantic_duplicates_from_pipe_separated(desc, model, similarity_threshold=0.85):
    if not desc or pd.isna(desc):
        return desc
    desc_list = desc.split("|")
    desc_list = sorted(desc_list, key=len, reverse=True)
    embeddings = model.encode(desc_list, convert_to_tensor=True)
    keep = [True] * len(desc_list)
    for i in range(len(desc_list)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(desc_list)):
            if not keep[j]:
                continue
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if sim > similarity_threshold:
                keep[j] = False
    filtered = [desc_list[i] for i in range(len(desc_list)) if keep[i]]
    return "|".join(filtered)


def map_by_substring(row_id, tax_df):
    matches = tax_df[tax_df["id"].apply(lambda tid: tid in str(row_id))]
    if not matches.empty:
        match = matches.iloc[0]
        return match["taxid"], match["rank"], match["name"]
    else:
        return pd.NA, pd.NA, pd.NA
    

def main():
    parser = argparse.ArgumentParser(description="Preprocess retrieved texts and map to original data.")
    parser.add_argument("--input", "-i", required=True, help="Path to input InterProScan or LCA .tsv file")
    parser.add_argument("--data", "-d", required=True, help="Path to test data (.fasta)")
    parser.add_argument("--type", "-t", default="lca", required=True, help="'interproscan' or 'lca'")
    parser.add_argument("--output", "-o", required=True, help="Path to output .csv file")
    parser.add_argument("--similarity_threshold", type=float, default=0.85, help="Semantic similarity threshold (default: 0.85)")

    args = parser.parse_args()
    
    df = fasta_to_df(args.data)

    if args.type == "interproscan":
        print(f"Processing {args.input}...")
        scan_df = parse_interproscan_file(args.input)
        # iscan.to_csv(args.output, index=False)
        # print(f"Saved to {args.output} (shape: {df.shape})")

        # Build description / map (hardcoded standard columns)
        desc_cols = ["Signature Description", "InterPro Description"]
        description_dict = build_interpro_description(scan_df, "id", desc_cols)
        df["desc"] = df["id"].map(lambda x: description_dict.get(str(x), ""))
    
        # Load model and remove duplicates
        model = SentenceTransformer("all-MiniLM-L6-v2")
        tqdm.pandas()
        df["desc_nodup"] = df["desc"].progress_apply(
            lambda x: remove_semantic_duplicates_from_pipe_separated(x, model, args.similarity_threshold)
        )
        # Remove duplicates after processing (keep first row per protein ID/MD5) -> take only descriptions
        df_unique = df.drop_duplicates(subset=["id"], keep="first")[["id", "desc", "desc_nodup"]]
        # Save
        df_unique.to_csv(args.output, index=False)
        print(f"Processed data (unique proteins: {len(df_unique)}) saved to {args.output}")


    elif args.type == "lca":
        tax_df = parse_mmseqs_lca(args.input)
        # tax_df.to_csv(args.output, index=False)
        # print(f"Parsed {len(tax_df)} records from {args.input} to {args.output}")

        # Map taxonomy
        tax_cols = df["id"].apply(lambda x: map_by_substring(x, tax_df))
        df["taxid"] = tax_cols.apply(lambda x: x[0])
        df["rank"] = tax_cols.apply(lambda x: x[1])
        df["name"] = tax_cols.apply(lambda x: x[2])

        # Save
        df.to_csv(args.output, index=False)
        print(f"Mapped taxonomy for {len(df)} rows to {args.output}")
        print(f"Taxonomy coverage: {df['taxid'].notna().sum()}/{len(df)} ({df['taxid'].notna().mean():.1%})")

    else:
        ValueError("Data type must be either 'interproscan' or 'lca'")

if __name__ == "__main__":
    main()
    