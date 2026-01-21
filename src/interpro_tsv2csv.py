import argparse
import pandas as pd

def parse_interproscan_file(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', header=None, comment='#')
    
    # Set column names according to InterProScan standard [web:7]
    df.columns = [
        'id',      # 0
        'Sequence MD5 Digest',    # 1
        'Sequence Length',        # 2
        'Analysis',               # 3
        'Signature Accession',    # 4
        'Signature Description',  # 5
        'Start',                  # 6
        'Stop',                   # 7
        'Score',                  # 8
        'Status',                 # 9
        'Date',                   # 10
        'InterPro Accession',     # 11
        'InterPro Description',   # 12
        'GO Annotations',         # 13
        'Pathway Annotations'     # 14
    ]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse InterProScan TSV and save as CSV.")
    parser.add_argument("--tsv_file", help="Path to input InterProScan TSV file")
    parser.add_argument("--output_file", help="Path to output CSV file")
    args = parser.parse_args()
    
    print(f"Processing {args.tsv_file}...")
    df = parse_interproscan_file(args.tsv_file)
    df.to_csv(args.output_file, index=False)
    print(f"Saved to {args.output_file} (shape: {df.shape})")
