import argparse
import pandas as pd

def parse_mmseqs_lca(filepath):
    records = []
    last_id = None
    with open(filepath, encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line: continue
            if '\t' not in line:
                # ID-only line, e.g. VFG029903
                last_id = line
            else:
                parts = line.split('\t')
                # If first part is an ID (not empty), use it
                if parts[0] and not parts[0].isspace():
                    rec_id = parts[0]
                    taxid, rank, name = parts[1:]
                else:
                    # First part empty: use last saved ID
                    rec_id = last_id
                    taxid, rank, name = parts[1:]
                records.append({
                    'id': rec_id,
                    'taxid': taxid,
                    'rank': rank,
                    'name': name,
                })
                last_id = None
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description='Parse MMseqs LCA TSV file into CSV')
    parser.add_argument('--input_file', help='Path to input LCA TSV file (e.g., alnRes_lca.tsv)')
    parser.add_argument('--output_file', '-o', default='alnRes_lca.csv', help='Path to output CSV file (default: alnRes_lca.csv)')
    args = parser.parse_args()

    tax_df = parse_mmseqs_lca(args.input_file)
    tax_df.to_csv(args.output_file, index=False)
    print(f"Parsed {len(tax_df)} records from {args.input_file} to {args.output_file}")

if __name__ == "__main__":
    main()

