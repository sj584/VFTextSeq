# VFTextSeq <br/>

<!-- ![VFTextSeq-viz](./vf.png) -->





## 🚀 Environment setup
```bash
# Clone repository
git clone https://github.com/sj584/VFTextSeq.git
cd VFTextSeq

# Create conda environment
conda create -n VFTextSeq python=3.11 -y
conda activate VFTextSeq

# install pytorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# install other libraries using requirements.txt 
pip install -r requirements.txt
python -m pip install git+https://github.com/PickyBinders/tea.git
```

<br/>

## Data details

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
│   ├── VFTextSeq_model.joblib                   # XGB model weight
|   ├── VFTextSeq_predictions.csv                # Prediction result 
│   ├── alnRes_lca_gtdb.tsv                      # mmseq_taxonomy result
│   ├── df_interproscan.csv                      # interproscan result per protein id
│   ├── df_interproscan_no_dup_semantic.csv      # interproscan with semantic deduplication
│   ├── df_taxonomy_gtdb.csv                     # mmseq_taxonomy result
│   ├── test.csv                                 # test with label
│   ├── train.csv                                # train with label
│   ├── test_best_hit_label_transfer.csv         # test with TEA-MMseqs2 transfered label
│   ├── train_best_hit_noself_label_transfer.csv # train with TEA-MMseqs2 transfered label
│   └── Interproscan_DeepVF_output.tsv           # interproscan original result   
│
└── VirulentHunter/                          # VirulentHunter benchmark and features
    ├── VFTextSeq_model.joblib                   # XGB model weight
    ├── VFTextSeq_predictions.csv                # Prediction result 
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
    ├── test_best_hit_label_transfer.csv         # test with TEA-MMseqs2 transfered label
    ├── trainval_best_hit_noself_label_transfer.csv # train with TEA-MMseqs2 transfered label
    └── virulent_output.tsv                      # interproscan original result
```

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
# 1-1. install InterProScan following instructions in https://github.com/ebi-pf-team/interproscan6
# 1-2. run interproscan to get annotations (~1 minute)
nextflow run ebi-pf-team/interproscan6   -r 6.0.2.2   -profile docker   --datadir data   --input example.fasta
# example.fasta.tsv: Tabular summary of InterPro matches

# 2. MMseqs2 Taxonomy
# 2.1. install mmseqs2
conda install -c conda-forge -c bioconda mmseqs2
# 2-2. download GTDB database (1~2 days)
mmseqs database GTDB mmseqs_gtdb/gtdb tmp
# 2-3. run taxonomy search on the database (several hours)
mmseqs easy-taxonomy example/example.fasta mmseq_gtdb/gtdb alnRes tmp
# alnRes_lca.tsv: Tabular summary of mmseqs taxonomy matches

# 3. TEA-MMseqs2
# 3-1. generate TEA tokens for train+val data
cat data/VirulentHunter/{train,val}.fasta > data/VirulentHunter/train+val.fasta
tea_convert \
  -f 'data/VirulentHunter/train+val.fasta' \
  -o 'data/VirulentHunter/train+val_tea.fasta' \
  -H
tea_convert \
  -f 'example/example.fasta' \
  -o 'data/VirulentHunter/example_tea.fasta' \
  -H
# 3-2. perform TEA-MMseqs2 search
TEA_MATRIX="$(python -c 'from importlib.resources import files; print(files("tea") / "matcha.out")')"
mmseqs easy-search \
    'data/VirulentHunter/example_tea.fasta' \
    'data/VirulentHunter/train+val_tea.fasta' \
    'data/VirulentHunter/tea_results.m8' \
    'data/VirulentHunter/tmp/' \
    --comp-bias-corr 0 \
    --mask 0 \
    --gap-open 18 \
    --gap-extend 3 \
    --sub-mat "$TEA_MATRIX" \
    --seed-sub-mat "$TEA_MATRIX" \
    --exact-kmer-matching 1
# tea_results.m8: TEA search result

# 4. Merge into single input csv
python src/preprocess.py \
    -i example/example.fasta \
    -ip example/example.fasta.tsv \
    -mp example/alnRes_lca.tsv \
    -tp data/VirulentHunter/tea_results.m8

# final preprocessed data saved to example/example_preprocessed.csv
```
<br>

### 2. Generate embedding and predict!
```bash
# 1. Generate embeddings
python src/extract_embedding.py -i example/example_preprocessed.csv

#2. Predict
python src/predict.py \
     -i example/example_preprocessed.csv \
     -e example/embedding \
     --model_path data/VirulentHunter/VFTextSeq_model.joblib
```

## 📚 References
1. ESM2 - Protein language model [GitHub](https://github.com/facebookresearch/esm)
2. InterProScan - Functional annotations [Github](https://github.com/ebi-pf-team/interproscan6)
3. MMseqs2 taxonomy - Taxonomy search [GitHub](https://github.com/soedinglab/MMseqs2) & [Document](https://github.com/soedinglab/mmseqs2/wiki)<br/>
4. BioBERT - language mode for text embedding [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1)
5. TEA - rewriting protein alphabets with language models [TEA](https://github.com/PickyBinders/tea)
