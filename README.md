# Focal Cortical Dysplasia Classification in Drug-Resistant Epilepsy

## Overview
This project implements a machine learning pipeline to classify Focal Cortical Dysplasia (FCD) using structural T1-weighted MRI data. We utilize morphological and intensity asymmetry features across 96 brain regions.

## Methodology
1.  **Preprocessing**: Skull-stripping (ANTsPyNet) and non-linear registration (SyN) to MNI152 space.
2.  **Feature Extraction**: Calculation of mean and standard deviation intensity asymmetry between hemispheric counterparts using the Harvard-Oxford atlas.
3.  **Classification**: Comparison of Random Forest, SVM, and L1-regularized Logistic Regression.
4.  **Validation**: Leave-One-Out Cross-Validation (LOOCV) and permutation testing (100 iterations).

## Key Results
*   **Best Model**: Logistic Regression (L1) achieved **78.0% accuracy** (p=0.02).
*   **Key Features**: Asymmetry in the Inferior Frontal Gyrus and Temporal Pole were top predictors.

## Dataset
Data provided by OpenNeuro: [ds004199](https://openneuro.org/datasets/ds004199)
