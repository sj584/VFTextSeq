# VFTextSeq <br/>

<!-- ![VFTextSeq-viz](./vf.png) -->

## Instruction
1. data
2. example
3. src

### Data directory

```text
data/
├── Case Study/                              # Species-specific case-study datasets
│   ├── SA_CDHIT_90.csv                          # S.aureus
│   ├── SP_CDHIT_90.csv                          # S.pneumoniae
│   ├── TB_CDHIT_90.csv                          # M.tuberculosis
│   ├── VC_CDHIT_90.csv                          # V.cholerae
│   └── YP_CDHIT_90.csv                          # Y.pestis
│
├── DeepVF/                                  # DeepVF benchmark and features
│   ├── DeepVF_Independent_Dataset/              # pos/neg fasta file
│   ├── DeepVF_Training_Dataset/                 # pos/neg fasta file
│   ├── VFTextSeq_model.pkl                      # XGB model weight
│   ├── alnRes_lca_gtdb.tsv                      # mmseq_taxonomy result
│   ├── df_interproscan.csv                      # interproscan result per protein id
│   ├── df_interproscan_no_dup_semantic.csv      # interproscan with semantic deduplication
│   ├── df_taxonomy_gtdb.csv                     # mmseq_taxonomy result
│   ├── test.csv                                 # test with label
│   └── train.csv                                # train with label
│
└── VirulentHunter/                          # VirulentHunter-style benchmark and features
    ├── VFTextSeq_model.pkl                      # XGB model weight
    ├── alnRes_lca_gtdb.tsv                      # mmseq_taxonomy result
    ├── df_interproscan.csv                      # interproscan result per protein id
    ├── df_interproscan_no_dup_semantic.csv      # interproscan with semantic deduplication
    ├── df_taxonomy_gtdb.csv                     # mmseq_taxonomy result
    ├── train.csv                                # train with detailed annotation
    ├── train.fasta                              # train fasta file
    ├── train_labels.csv                         # train with label
    ├── val.csv                                  # val with detailed annotation
    ├── val.fasta                                # val fasta file
    ├── val_labels.csv                           # val with label
    ├── test.csv                                 # test with detailed annotation
    ├── test.fasta                               # test with label
    ├── test_labels.csv                          # test with label
    └── virulent_output.tsv                      # interproscan original result

## 🚀 Environment setup
```bash
# Clone repository
git clone https://github.com/sj584/VFTextSeq.git
cd VFTextSeq

# Create conda environment
conda create -n VFTextSeq python=3.10
conda activate VFTextSeq

# install pytorch & pip
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pip -y

# install other libraries using requirements.txt 
pip install -r requirements.txt
```

<br/>

## 📁 Input Format

### Example FASTA 

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

<br/>

### 1. Preprocess input data
```bash
# 1. InterProScan
# 1.1 run interproscan to get annotations (several hours)
./interproscan.sh -i example.fasta -f tsv -o example_interproscan.tsv

# 2. MMseqs2 Taxonomy
# 2-1. load GTDB database (1~2 days)
mmseqs database GTDB mmseqs_gtdb/gtdb tmp
# 2-2. run taxonomy search on the database (several hours)
mmseqs easy-taxonomy example.fasta mmseq_gtdb/gtdb alnRes tmp

# 3. preprocess text data (.tsv files)
# 3-1. convert tsv into csv
# 3-2. remove redundant texts (interproscan) from the annotations
python src/preprocess.py -i example/example.fasta -ip example/example_interproscan.tsv -mp example/alnRes_lca.tsv
# outputs are saved at the same directory as the input fasta file
# e.g., example/example_preprocessed.csv
```

<br/>

### 2. Extract embeddings

```bash
python src/extract_embedding.py -i example/example_preprocessed.csv
# By default, esm2-650M and bert models are used
# outputs are saved under embedding/ directory under the same directory as the input file
# e.g., example/embedding/*.npy
```

<br/>

# Prediction

```bash
python src/predict.py -i example/example_preprocessed.csv -e example/embedding
```

<br/>

## 📁 Model Prediction Results
### Example CSV
<br/>

| id                | prob | pred |
|-------------------|-------------|-----------|
| sp\|P26683\|SIGA_NOSS1 | 0.2253     | 0         |
| VFG007156         | 0.9942     | 1         | 
| VFG007971         | 0.9359     | 1         |

<br/>

## 📚 References
1. ESM2 - Protein language model [GitHub](https://github.com/facebookresearch/esm)
2. InterProScan - Functional annotations [Document](https://interproscan-docs.readthedocs.io/en/v5/#)
3. MMseqs2 taxonomy - Taxonomy search [GitHub](https://github.com/soedinglab/MMseqs2) & [Document](https://github.com/soedinglab/mmseqs2/wiki)<br/>
4. BERT - language mode for text embedding [BERT](https://huggingface.co/google-bert/bert-base-uncased) 
