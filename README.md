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

# install pytorch & pip
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pip -y

# install other libraries using requirements.txt 
pip install -r requirements.txt
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
# 1.1 run interproscan to get annotations (several hours)
./interproscan.sh -i example.fasta -f tsv -o example_interproscan.tsv

# 2. MMseqs2 Taxonomy
# 2-1. load GTDB database (1~2 days)
mmseqs database GTDB mmseqs_gtdb/gtdb tmp
# 2-2. run taxonomy search on the database (several hours)
mmseqs easy-taxonomy example.fasta mmseq_gtdb/gtdb alnRes tmp

# 3. TEA-MMseqs2
# 3-1. generate TEA tokens
python tea_fasta.py VirulentHunter/train_val.fasta VirulentHunter/trainval_tea.fasta --batch_size 16
python tea_fasta.py example.fasta example_tea.fasta --batch_size 16
# 3-2. perform TEA-MMseqs2 search
mmseqs easy-search example_tea.fasta VirulentHunter/trainval_tea.fasta VirulentHunter/results.m8 VirulentHunter/tmp/ \
    --comp-bias-corr 0 \
    --mask 0 \
    --gap-open 18 \
    --gap-extend 3 \
    --sub-mat /home/user/miniconda3/envs/TEA/lib/python3.11/site-packages/tea/matcha.out \
    --seed-sub-mat /home/user/miniconda3/envs/TEA/lib/python3.11/site-packages/tea/matcha.out \
    --exact-kmer-matching 1
```

## 📚 References
1. ESM2 - Protein language model [GitHub](https://github.com/facebookresearch/esm)
2. InterProScan - Functional annotations [Document](https://interproscan-docs.readthedocs.io/en/v5/#)
3. MMseqs2 taxonomy - Taxonomy search [GitHub](https://github.com/soedinglab/MMseqs2) & [Document](https://github.com/soedinglab/mmseqs2/wiki)<br/>
4. BioBERT - language mode for text embedding [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1)
5. TEA - rewriting protein alphabets with language models [TEA](https://github.com/PickyBinders/tea)
