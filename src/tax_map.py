import argparse
import pandas as pd

def map_by_substring(row_id, tax_df):
    matches = tax_df[tax_df['id'].apply(lambda tid: tid in str(row_id))]
    if not matches.empty:
        match = matches.iloc[0]
        return match['taxid'], match['rank'], match['name']
    else:
        return pd.NA, pd.NA, pd.NA

def main():
    parser = argparse.ArgumentParser(description='Map taxonomy from LCA CSV to input CSV by substring ID matching')
    parser.add_argument('--lca_file', '-l', required=True, help='LCA CSV file (e.g., alnRes_lca.csv)')
    parser.add_argument('--input_file', '-i', required=True, help='Input CSV file with "id" column (e.g., example.csv)')
    parser.add_argument('--output_file', '-o', required=True, help='Output CSV with added taxid, rank, name columns')
    args = parser.parse_args()

    # Load data
    tax_df = pd.read_csv(args.lca_file)
    df_input = pd.read_csv(args.input_file)

    # Map taxonomy
    tax_cols = df_input['id'].apply(lambda x: map_by_substring(x, tax_df))
    df_input['taxid'] = tax_cols.apply(lambda x: x[0])
    df_input['rank'] = tax_cols.apply(lambda x: x[1])
    df_input['name'] = tax_cols.apply(lambda x: x[2])

    # Save
    df_input.to_csv(args.output_file, index=False)
    print(f"Mapped taxonomy for {len(df_input)} rows to {args.output_file}")
    print(f"Taxonomy coverage: {df_input['taxid'].notna().sum()}/{len(df_input)} ({df_input['taxid'].notna().mean():.1%})")

if __name__ == "__main__":
    main()

