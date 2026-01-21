# VFTextSeq

# Environment setting
```python
# conda environment creation
conda create -n VFTextSeq

# conda activate VFTextSeq environment
conda activate VFTextSeq

### installation
### you can use any library versions according to your system environment
# pip install
conda install pip -y
# install using requirements.txt (pytorch to avoid version)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

# Data processing steps

1. ESM2 embedding      [ESM2 GitHub](https://github.com/facebookresearch/esm)
3. InterProScan      [InterProScan Document](https://interproscan-docs.readthedocs.io/en/v5/#)
4. MMseqs2 taxonomy      [MMseqs2 GitHub](https://github.com/soedinglab/MMseqs2) & [MMseqs2 Document](https://github.com/soedinglab/mmseqs2/wiki)


ESM2 embedding (650M)
```python
python esm_embedding.py --fasta_path example.fasta --output_dir example/esm_emb
```

InterProScan
```python
# run interproscan to get annotations (several hours)
./interproscan.sh -i example.fasta -f tsv -o example_interproscan.tsv

# change tsv file into csv for column annotation
python interpro_tsv2csv.py --tsv_file example/example.csv --output_file example/example_interproscan.csv

# run semantic removal for interproscan
python interpro_remove_semantics.py --csv_file example/example.csv --interpro_csv example/example_interproscan.csv --output_file example/example_interproscan_rm_dup.csv

# get bert embedding of interproscan
python interpro_Bert_emb.py --input_csv example/example_interproscan_rm_dup.csv --output_dir example/interproscan_bert_emb

```

MMseqs2 taxonomy
```python
# load GTDB database (1~2 days)
mmseqs database GTDB mmseqs_gtdb/gtdb tmp

# run taxonomy search on the database (several hours)
mmseqs easy-taxonomy example.fasta mmseq_gtdb/gtdb alnRes tmp

# change tsv file into csv file (add column)
python tax_tsv2csv --input_file alnRes.tsv --output_file alnRes.csv

# get bert embedding of interproscan
python interpro_Bert_emb.py --input_csv example/alnRes.csv --output_dir example/tax_bert_emb
```
