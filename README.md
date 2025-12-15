# 💳 Fraud Anomaly Detection

An end-to-end machine learning and deep learning project that detects fraudulent credit-card transactions using **supervised models** (Logistic Regression, Random Forest, XGBoost) and **unsupervised anomaly detection models** (Simple Autoencoder, Deep Autoencoder).

This project explores fraud patterns, handles extreme class imbalance, evaluates multiple modeling strategies, and includes a **Streamlit app** for model comparison and transaction-level fraud scoring.

---

## Project Structure

```
FraudAnomalyDetection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── log_reg_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── autoencoder_model.pth
│   └── deep_autoencoder_model.pth
│
├── src/
|   ├──data_loader.py
│   ├── preprocessing.py
|   ├── models_supervised.py
│   ├── train_supervised.py
|   ├── models_autoencoders.py
│   ├── train_autoencoders.py
│   ├── evaluate_autoencoder.py
|   └── utils.py
|
├── notebooks/
|   ├──FraudDetectionAnomaly.ipynb
|
├── requirements.txt 
├── app.py
└── README.md
```

---

# Project Overview

Credit-card fraud datasets are highly imbalanced — fraud represents **only 0.17%** of all transactions.
This project compares:

###  **Supervised Models**

Trained on labeled fraud vs non-fraud data using class-weighting and resampling strategies:

* Logistic Regression (class-weighted)
* Random Forest (class-weighted)
* XGBoost (scale_pos_weight)

###  **Imbalance Handling**

* Class Weighting
* SMOTE Oversampling
* SMOTE-ENN (Oversampling + Noise Removal)

###  **Unsupervised Models (Anomaly Detection)**

Trained only on *normal* transactions:

* Simple Autoencoder
* Deep Autoencoder
* Threshold selected using **ROC analysis (Youden’s J statistic)**

  * Best threshold found: **0.4871**

---

##  Exploratory Data Analysis

Key findings:

* Fraud = **0.172%** of data (highly imbalanced)
* PCA visualization shows slight separation but still overlapping regions
* t-SNE visualization reveals distinct fraud clusters in 2D space
* Several features (V14, V12, V4) have stronger correlation with fraud

---
## Model Performance Comparison
Supervised Models (Class-Weighted)

| Model                   | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | Notes                                        |
| ----------------------- | ----------------- | -------------- | ---------------- | -------------------------------------------- |
| **Logistic Regression** | 0.0609            | **0.9184**     | 0.1142           | Very high recall but extremely low precision |
| **Random Forest**       | **0.9487**        | 0.7551         | 0.8409           | Best precision among supervised models       |
| **XGBoost**             | 0.8400            | **0.8571**     | **0.8485**       | Best overall supervised performance          |

Oversampling Models (SMOTE / SMOTE-ENN)

| Model                   | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | Notes                                          |
| ----------------------- | ----------------- | -------------- | ---------------- | ---------------------------------------------- |
| **XGBoost + SMOTE**     | 0.3755            | **0.8776**     | 0.5260           | High recall but many false positives           |
| **XGBoost + SMOTE-ENN** | 0.3484            | **0.8673**     | 0.4971           | Similar to SMOTE with slightly lower precision |

Unsupervised Models (Autoencoders)

| Model                  | Precision (Fraud)                        | Recall (Fraud) | F1-Score (Fraud) | Additional Metrics                      |
| ---------------------- | ---------------------------------------- | -------------- | ---------------- | --------------------------------------- |
| **Simple Autoencoder** | *(Not explicitly evaluated in notebook)* | —              | —                | Separation visible in RE histogram      |
| **Deep Autoencoder**   | 0.0318                                   | **0.8537**     | 0.0614           | **AP = 0.4327**, Threshold = **0.4871** |

# Observations

**Best Supervised Model:** XGBoost (best balance of precision & recall)

**Best Precision:** Random Forest

**Best Recall:** Logistic Regression (but at cost of precision)

**Autoencoder:** Excellent recall but low precision (typical for anomaly detection)

**Oversampling:** Increases recall, reduces precision (more false positives)

# Error Analysis

Manually computed TP, FP, FN, TN:

* **True Positives:** 420
* **False Positives:** 12778
* **False Negatives:** 72
* **True Negatives:** 271537

  
<p align="center">
  <img src="/assets/pca_plot.png" alt="PCA PROJECTION PLOT" width="550">
</p>


Additional insights:

* FN errors overlap heavily with TP errors
* Fraud samples missed (FN) tend to have **lower reconstruction error**, near threshold
* TP and FN error distributions compared with histograms & boxplots

---

# Streamlit Application

The app allows users to:

### Upload CSV transactions

### Select a model:

* Logistic Regression
* Random Forest
* XGBoost
* Autoencoder
* Deep Autoencoder

###  View:

* Fraud predictions
* Fraud severity meter
* Download results as CSV

A clean UI with descriptions for each model is included.

---

# Saved Models

All trained models are stored in `/models/`:

```
log_reg_model.pkl
rf_model.pkl
xgb_model.pkl
autoencoder_model.pth
deep_autoencoder_model.pth
```

---

#  Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-Learn
* PyTorch
* Streamlit

---

# Summary

This project provides:

✔ Full fraud-detection pipeline
✔ Class imbalance handling
✔ Supervised vs. unsupervised comparison
✔ Deep learning autoencoders
✔ Threshold selection via ROC analysis
✔ Error analysis (TP, FP, FN, TN)
✔ Interactive Streamlit dashboard
✔ Saved production-ready ML and AE models

---

