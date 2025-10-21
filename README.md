# Machine Learning Model predicting BIS from single-channel EEG signals

## Overview  
This repository provides the full implementation of a deep neural network designed to predict the **Bispectral Index (BIS)** — a quantitative measure of anesthetic depth — from EEG data. The study compares **raw EEG inputs** with **EEMD-decomposed signals**, revealing that raw EEG preserves more informative features for accurate BIS prediction.

## Model Architecture  
The final predictive pipeline consists of a **CNN–Transformer–BiLSTM ensemble** with a **Gradient Boosted Regression Tree (GBRT)** meta-model.  
Training and evaluation are performed in `Final_Regression.py`.

## Dataset  
Both the training and validation datasets were obtained from the **VitalDB** database. Preprocessing and filtering details are documented in `VitalDBDataset.py`.

## Experiments  
This repository includes experiment scripts and analyses detailing:  
- Iterations of models trained and validated on **EEMD-decomposed EEG** (`EEMD_Model_Iterations.py`)  
- Iterations of models trained and validated on **raw EEG** (`Raw_Model_Iterations.py`)  
- Data distribution visualizations and exploratory analysis (`Analyze_Dataset.py`)

