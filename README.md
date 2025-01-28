# Autism Risk Gene Prediction Using Graph and Sequence Embeddings

This repository contains code and instructions for reproducing the analysis conducted in the MSc thesis, *"Predicting Autism Risk Genes Using Graph and Sequence Embeddings"*. The study utilized biological datasets, protein-protein interaction graphs, and DNA/protein sequences to develop machine learning models for identifying Autism risk genes.

## Table of Contents

1. [Repository Structure](#repository-structure) 
2. [Data Preparation](#data-preparation)  
3. [Embedding Creation](#embedding-creation)  
4. [Machine Learning Models](#machine-learning-models)  
5. [Validation and Results](#validation-and-results)  


## Dataset Preparation

The Data Preparation stage consists in the extraction of the postive and negative gene datasets


#### Postive genes:
SFARI gene dataset 

#### Negative genes:
Updated list from the Krishnan et al. [article](https://www.nature.com/articles/nn.4353)


---

## Graph and Sequence Embedding Generation

The embeddings were created using third-party tools and custom scripts:  

1. **Graph Embeddings**:

2. **Sequence Embeddings**:



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