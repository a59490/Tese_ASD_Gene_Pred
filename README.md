# Autism Risk Gene Prediction Using Graph and Sequence Embeddings

This repository contains code and instructions for reproducing the analysis conducted in the MSc thesis, *"Predicting Autism Risk Genes Using Graph and Sequence Embeddings"*. The study utilized biological datasets, protein-protein interaction graphs, and DNA/protein sequences to develop machine learning models for identifying Autism risk genes.
Three embedding aproaches were used, using DNA sequences [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2/tree/main), Protein sequences [ProtT5](https://github.com/agemagician/ProtTrans) and Graphs [GRAPE](https://github.com/AnacletoLAB/grape)

## Table of Contents

1. [Repository Structure](#repository-structure) 
2. [Data Preparation](#data-preparation)  
3. [Embedding Creation](#embedding-creation)  
4. [Machine Learning Models](#machine-learning-models)  
5. [Validation and Results](#validation-and-results)  


## Dataset Preparation

The Data Preparation stage consists in the extraction of the postive and negative gene datasets


#### Postive genes:
[SFARI](https://gene.sfari.org/database/human-gene/) gene dataset version 01/16/2024 

#### Negative genes:
Updated list from the Krishnan et al. [article](https://www.nature.com/articles/nn.4353)


---

## Graph and Sequence Embedding Generation

Two diferent embedding aproaches are presented. 

### Graph Embeddings:
In order to create the Graph embeddings you will need:

- *STRINGdb PiP graph*
- *Sfari gene dataset*
- *Positive and negative gene list*

To create the embeddings run the script:

    # Create_embs.py

This scrip will try to create an embedding for each of the available embedding algoriths available in the [GRAPE](https://github.com/AnacletoLAB/grape) library

### Sequence Embeddings:



---

## Model Training and Evaluation

Six machine learning models were trained and evaluated:

- Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, and KNN.  

### Steps:
1. **Prepare Train, Validation, and Test Sets**:

2. **Train Models**:

3. **Evaluate Models**:


---

## Results Validation