# VFTextSeq
<br/><br/>
# Environment (version compatibility O)
```python
### conda env setting ###

# download and locate to the github repository
git clone https://github.com/sj584/VFTextSeq.git
cd VFTextSeq

# conda environment creation
conda create -n VFTextSeq

# conda activate VFTextSeq environment
conda activate VFTextSeq

# install pip
conda install pip -y

# install using requirements.txt (url for pytorch version)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```
<br/><br/>
# Data processing steps for generating embeddings

1. ESM2      [ESM2 GitHub](https://github.com/facebookresearch/esm)
2. InterProScan      [InterProScan Document](https://interproscan-docs.readthedocs.io/en/v5/#)
3. MMseqs2 taxonomy      [MMseqs2 GitHub](https://github.com/soedinglab/MMseqs2) & [MMseqs2 Document](https://github.com/soedinglab/mmseqs2/wiki)
[BERT](https://huggingface.co/google-bert/bert-base-uncased)

## Example csv Input

| id              | label |
|-----------------|-------|
| sp&#124;P26683&#124;SIGA_NOSS1 | 0     |
| VFG007156       | 1     |
| VFG007971       | 1     |

## Example FASTA Input

```fasta
>sp|P26683|SIGA_NOSS1
MNQANNVLDSIYQPDLEIMNQPEIELDDLLIEEDEDLLLADDGDIDEFLEPQTDEDDAKSGKAAKSRRRTQSKKKHYTEDSIRLYLQEIGRIRLLRADEEIELARKIADLLELERVRERLSEKLERDPRDSEWAEAVQLPLPAFRYRLHIGRRAKDKMVQSNLRLVVSIAKKYMNRGLSFQDLIQEGSLGLIRAAEKFDHEKGYKFSTYATWWIRQAITRAIADQSRTIRLPVHLYETISRIKKTTKLLSQEMGRKPTEEEIATRMEMTIEKLRFIAKSAQLPISLETPIGKEEDSRLGDFIESDGETPEDQVSKNLLREDLEKVLDSLSPRERDVLRLRYGLDDGRMKTLEEIGQIFNVTRERIRQIEAKALRKLRHPNRNSVLKEYIR
>VFG007156
MAYQASDLMADVIALVEQRWVSSEEIWKIATSMELVAIEQKIDFFRELHKLIRHIPVDVFADDEQRQNLIQAAQKALDEAIDLEEEEAWDDELD
>VFG007971
MAFTRIHSFLASAGNTSMYKRVWRFWYPLMTHKLGTDEIMFINWAYEEDPPMALPLEASDEPNRAHINLYHRTATQVNLSGKRILEVSCGHGGGASYLTRALHPASYTGLDLNPAGIKLCQKRHQLPGLEFVRGDAENLPFDNESFDVVINIEASHCYPHFPRFLAEVVRVLRPGGHLAYADLRPSNKVGEWEVDFANSRLQQLSQREINAEVLRGIASNSQKSRDLVDRHLPAFLRFAGREFIGVQGTQLSRYLEGGELSYRMYSFAKD
```

## [Note]<br> **when generating embeddings. <br>embedding file should be saved as {id}.pt**

## ESM2 embedding (650M)
```python
python src/esm_embedding.py --fasta_path example/example.fasta --output_dir example/esm_emb
```

## InterProScan
```python
# run interproscan to get annotations (several hours)
./interproscan.sh -i example.fasta -f tsv -o example_interproscan.tsv

# change tsv file into csv for column annotation
python src/interpro_tsv2csv.py --tsv_file example/example_interproscan.tsv --output_file example/example_interproscan.csv

# run semantic removal for interproscan
python src/interpro_remove_semantics.py --csv_file example/example.csv --interpro_csv example/example_interproscan.csv --output_file example/example_interproscan_rm_dup.csv

# get bert embedding of interproscan
python src/interpro_Bert_emb.py --input_csv example/example_interproscan_rm_dup.csv --output_dir example/interproscan_bert_emb

```

## MMseqs2 taxonomy
```python
# load GTDB database (1~2 days)
mmseqs database GTDB mmseqs_gtdb/gtdb tmp

# run taxonomy search on the database (several hours)
mmseqs easy-taxonomy example.fasta mmseq_gtdb/gtdb alnRes tmp

# change tsv file into csv file (add column)
python src/tax_tsv2csv.py --input_file example/alnRes_lca.tsv --output_file example/alnRes_lca.csv

# map taxonomy to input csv file
python src/tax_map.py --lca_file example/alnRes_lca.csv --input_file example/example.csv --output_file example/example_taxonomy.csv

# get bert embedding of interproscan
python src/tax_Bert_emb.py --input_csv example/example_taxonomy.csv --output_dir example/tax_bert_emb
```


# Prediction
```python
python src/predict.py --input_csv example/example.csv --output result.csv --esm_emb example/esm_emb --interproscan_emb example/interproscan_bert_emb --tax_emb example/tax_bert_emb
```
