# VFTextSeq <br/>

<!-- ![VFTextSeq-viz](./vf.png) -->

## 🚀 Environment setup
```bash
# Clone repository
git clone https://github.com/sj584/VFTextSeq.git
cd VFTextSeq

# Create conda environment
conda create -n VFTextSeq python=3.10
conda activate VFTextSeq

# install pip & pytorch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# install other libraries using requirements.txt 
pip install -r requirements.txt
```

<br/>

## 📁 Input Format

### Example csv 

| id              | label |
|-----------------|-------|
| sp&#124;P26683&#124;SIGA_NOSS1 | 0     |
| VFG007156       | 1     |
| VFG007971       | 1     |

**Note:** **label column is not necessary -> Prediction**

## Example FASTA 

```fasta
>sp|P26683|SIGA_NOSS1
MNQANNVLDSIYQPDLEIMNQPEIELDDLLIEEDEDLLLADDGDIDEFLEPQTDEDDAKSGKAAKSRRRTQSKKKHYTEDSIRLYLQEIGRIRLLRADEEIELARKIADLLELERVRERLSEKLERDPRDSEWAEAVQLPLPAFRYRLHIGRRAKDKMVQSNLRLVVSIAKKYMNRGLSFQDLIQEGSLGLIRAAEKFDHEKGYKFSTYATWWIRQAITRAIADQSRTIRLPVHLYETISRIKKTTKLLSQEMGRKPTEEEIATRMEMTIEKLRFIAKSAQLPISLETPIGKEEDSRLGDFIESDGETPEDQVSKNLLREDLEKVLDSLSPRERDVLRLRYGLDDGRMKTLEEIGQIFNVTRERIRQIEAKALRKLRHPNRNSVLKEYIR
>VFG007156
MAYQASDLMADVIALVEQRWVSSEEIWKIATSMELVAIEQKIDFFRELHKLIRHIPVDVFADDEQRQNLIQAAQKALDEAIDLEEEEAWDDELD
>VFG007971
MAFTRIHSFLASAGNTSMYKRVWRFWYPLMTHKLGTDEIMFINWAYEEDPPMALPLEASDEPNRAHINLYHRTATQVNLSGKRILEVSCGHGGGASYLTRALHPASYTGLDLNPAGIKLCQKRHQLPGLEFVRGDAENLPFDNESFDVVINIEASHCYPHFPRFLAEVVRVLRPGGHLAYADLRPSNKVGEWEVDFANSRLQQLSQREINAEVLRGIASNSQKSRDLVDRHLPAFLRFAGREFIGVQGTQLSRYLEGGELSYRMYSFAKD
```

<br/><br/>
# 🧬Data processing steps for generating embeddings<br/>

**Note:** Embedding files must be saved as **{id}**.pt

<br/>

### 1. ESM2 Embeddings (650M)
```bash
python src/esm_embedding.py --fasta_path example/example.fasta --output_dir example/esm_emb
```

<br/>

### 2. InterProScan Annotations
```bash
# run interproscan to get annotations (several hours)
./interproscan.sh -i example.fasta -f tsv -o example_interproscan.tsv

# preprocess interproscan data
# 1. convert tsv into csv
# 2. Remove redundant texts from the annotations
python src/preprocess.py -d example/example.fasta -i example/example_interproscan.tsv -t "interproscan" -o example/example_interproscan.csv

# 3. get bert embedding of interproscan data
python src/bert_embedding.py -i example/example_interproscan.csv -t "interproscan" -o example/interproscan_bert_emb
```

<br/>

### 3. MMseqs2 Taxonomy

```bash
# load GTDB database (1~2 days)
mmseqs database GTDB mmseqs_gtdb/gtdb tmp

# run taxonomy search on the database (several hours)
mmseqs easy-taxonomy example.fasta mmseq_gtdb/gtdb alnRes tmp

# preprocess taxonomy data
# 1. convert tsv into csv
# 2. map the LCA annotations to the input 
python src/preprocess.py -d example/example.fasta -i example/alnRes_lca.tsv -t "lca" -o example/example_taxonomy.csv

# 3. get bert embedding of interproscan
python src/bert_embedding.py -i example/example_taxonomy.csv -t "lca" -o example/tax_bert_emb
```

<br/>

# Prediction
```bash
python src/predict.py --input_csv example/example.csv --output result.csv --esm_emb example/esm_emb --interproscan_emb example/interproscan_bert_emb --tax_emb example/tax_bert_emb
```

```bash
python src/predict.py --input_csv example/example_nolabel.csv --output result_nolabel.csv --esm_emb example/esm_emb --interproscan_emb example/interproscan_bert_emb --tax_emb example/tax_bert_emb
```

<br/>

## 📁 Model Prediction Results

| id                | prob | pred | label |
|-------------------|-------------|-----------|------------|
| sp\|P26683\|SIGA_NOSS1 | 0.0135     | 0         | 0          |
| VFG007156         | 0.9783     | 1         | 1          |
| VFG007971         | 0.9943     | 1         | 1          |

<br/>

| id                | prob | pred |
|-------------------|-------------|-----------|
| sp\|P26683\|SIGA_NOSS1 | 0.0135     | 0         |
| VFG007156         | 0.9783     | 1         | 
| VFG007971         | 0.9943     | 1         |

### 📚 References
1. ESM2 - Protein language model [GitHub](https://github.com/facebookresearch/esm)
2. InterProScan - Functional annotations [Document](https://interproscan-docs.readthedocs.io/en/v5/#)
3. MMseqs2 taxonomy - Taxonomy search [GitHub](https://github.com/soedinglab/MMseqs2) & [Document](https://github.com/soedinglab/mmseqs2/wiki)<br/>
4. BERT - language mode for text embedding [BERT](https://huggingface.co/google-bert/bert-base-uncased) 
