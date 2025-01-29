# Autism Risk Gene Prediction Using Graph and Sequence Embeddings

This repository contains code and instructions for reproducing the analysis conducted in the MSc thesis, *"Predicting Autism Risk Genes Using Graph and Sequence Embeddings"*. The study utilized biological datasets, protein-protein interaction (PPI) graphs, and DNA/protein sequences to develop machine learning models for identifying Autism risk genes. Three embedding approaches were used:

- **DNA sequences** ([DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2/tree/main))
- **Protein sequences** ([ProtT5](https://github.com/agemagician/ProtTrans))
- **Graphs** ([GRAPE](https://github.com/AnacletoLAB/grape))

---

## Table of Contents

1. [Dataset Preparation](#dataset-preparation)  
2. [Embedding Creation](#embedding-creation)  
   - [Graph Embeddings](#a-graph-embeddings)  
   - [Sequence Embeddings](#b-sequence-embeddings)  
3. [Machine Learning Models](#machine-learning-models)  
4. [Validation and Results](#validation-and-results)  

---


## Dataset Preparation

The **Data Preparation** stage consists of extracting positive and negative gene datasets.

### Positive Genes:
- **Source:** [SFARI Gene Database](https://gene.sfari.org/database/human-gene/) (Version: 01/16/2024)

### Negative Genes:
- **Source:** Updated list from the Krishnan et al. [article](https://www.nature.com/articles/nn.4353)

---

## Embedding Creation

This project employs two different embedding approaches: **Graph Embeddings** and **Sequence Embeddings**.

### A) Graph Embeddings

Graph embeddings are generated using the following resources:
- **STRINGdb PPI Graph**
- **SFARI Gene Dataset**
- **Positive and Negative Gene Lists**

#### Steps:

1. Navigate to the `02_Embedding_creation/graph/` folder.
2. Run the script to create embeddings using all available algorithms in the [GRAPE](https://github.com/AnacletoLAB/grape) library:

   ```bash
   python3 Create_embs.py
   ```

### B) Sequence Embeddings

#### DNA Embeddings:

To generate DNA sequence embeddings:

1. Download the **cDNA FASTA files** from the [Ensembl FTP](https://www.ensembl.org/info/data/ftp/index.html).
2. Create the necessary dataset by running the `dna_seq_maker` script from the `02_Embedding_creation` folder.
3. Follow the [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2/tree/main) setup guide to create the environment.
4. Navigate to `02_Embedding_creation/dnaBERT/` and run:

   ```bash
   python3 dna_embeddings.py
   ```

#### Protein Embeddings:

To generate protein sequence embeddings:

1. Download the **protein FASTA files** from the [Ensembl FTP](https://www.ensembl.org/info/data/ftp/index.html).
2. Create the necessary dataset by running the `prot_seq_maker` script from the `02_Embedding_creation` folder.
3. Follow the [ProtT5](https://github.com/agemagician/ProtTrans) setup guide to create the environment.
4. Navigate to `02_Embedding_creation/prott5/` and run:

   ```bash
   python3 prot_emb.py
   ```

---

## Machine Learning Models

The `fold_model_compare.py` script is used for:
- Hyperparameter tuning
- Training multiple machine learning models
- Evaluating performance on dataset permutations (depending on SFARI categories used)

#### Supported Models:
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- XGBoost
- LightGBM
- k-Nearest Neighbors (KNN)

### Running the Script:

- To train and evaluate **all models**:

   ```bash
   python3 fold_model_compare.py all
   ```

- To train and evaluate a **specific model** (e.g., Random Forest):

   ```bash
   python3 fold_model_compare.py rf
   ```

### Outputs:
- Results are saved in `Results/{model_name}/`.
- Metrics include AUC, accuracy, F1-score, sensitivity, specificity, MCC, and more.

---

## Validation and Results

### Ranked List

- Generate a ranked list of all available genes using the trained model.
- Apply the ranked list to evaluate gene relevance.

### Enrichment Analysis

- Divide the ranked list into deciles.
- Compare each decile with ASD phenotypes and other gene association studies.

### Network Analysis

- Create a PPI graph of the top-ranked decile.
- Perform decile-specific enrichment analysis.

---

This repository provides all the necessary tools for generating embeddings, training models, and performing enrichment analyses. For further questions or issues, feel free to contact the repository maintainers.

