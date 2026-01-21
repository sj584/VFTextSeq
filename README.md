# VFTextSeq

## environment
pytorch
numpy
transformers


# Data processing steps

1. ESM2 embedding      [ESM2 GitHub](https://github.com/facebookresearch/esm)
3. InterProScan      [InterProScan Document](https://interproscan-docs.readthedocs.io/en/v5/#)
4. MMseqs2 taxonomy      [MMseqs2 GitHub](https://github.com/soedinglab/MMseqs2) & [MMseqs2 Document](https://github.com/soedinglab/mmseqs2/wiki)


ESM2 embedding (650M)
```python
python esm_embedding.py -i example.fasta -out_dir esm_emb
```

InterProScan
```python
# run interproscan to get annotations (several hours)
./interproscan.sh -i example.fasta -f tsv -o example_interproscan.tsv
```

MMseqs2 taxonomy
```python
# load GTDB database (1~2 days)
mmseqs database GTDB mmseqs_gtdb/gtdb tmp

# run taxonomy search on the database (several hours)
mmseqs easy-taxonomy example.fasta mmseq_gtdb/gtdb alnRes tmp
```
