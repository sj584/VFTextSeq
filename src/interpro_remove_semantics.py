import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def build_interpro_description(df_inter, id_col, desc_cols):
    description_map = {}
    for entry, group in df_inter.groupby(id_col):
        descs = set()
        for col in desc_cols:
            if col in group:
                for val in group[col].dropna():
                    if val != '-':
                        descs.add(str(val))
        description_map[entry] = '|'.join(sorted(descs))
    return description_map

def remove_semantic_duplicates_from_pipe_separated(desc, model, similarity_threshold=0.85):
    if not desc or pd.isna(desc):
        return desc
    desc_list = desc.split('|')
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
    return '|'.join(filtered)

def main():
    parser = argparse.ArgumentParser(description='Process InterProScan CSV descriptions and remove semantic duplicates.')
    parser.add_argument('--csv_file', required=True, help='Path to input CSV file (df_total)')
    parser.add_argument('--interpro_csv', required=True, help='Path to InterProScan CSV file')
    parser.add_argument('--output_file', required=True, help='Path for final output CSV file')
    parser.add_argument('--similarity_threshold', type=float, default=0.85, help='Semantic similarity threshold (default: 0.85)')
    
    args = parser.parse_args()
    
    # Load df_total
    df_total = pd.read_csv(args.csv_file)
     
    # Load InterPro CSV
    df_inter = pd.read_csv(args.interpro_csv)
    
    # Build description map (hardcoded standard columns)
    desc_cols = ['Signature Description', 'InterPro Description']
    description_dict = build_interpro_description(df_inter, 'id', desc_cols)
    
    # Map to df_total (assumes 'id' column matches)
    df_total['desc'] = df_total['id'].map(lambda x: description_dict.get(str(x), ''))
    
    # Load model and remove duplicates
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tqdm.pandas()
    df_total['desc_nodup'] = df_total['desc'].progress_apply(
        lambda x: remove_semantic_duplicates_from_pipe_separated(x, model, args.similarity_threshold)
    )
    
    # Remove duplicates after processing (keep first row per protein ID/MD5) -> take only descriptions
    df_unique = df_total.drop_duplicates(subset=['id'], keep='first')[['id', 'desc', 'desc_nodup']]


    # Save
    df_unique.to_csv(args.output_file, index=False)
    print(f"Processed data (unique proteins: {len(df_unique)}) saved to {args.output_file}")


if __name__ == "__main__":
    main()
