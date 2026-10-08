# Diabetes Prediction using Machine Learning

A machine learning system for diabetes prediction based on two datasets: **PIMA (`diabetes.csv`)** and **Frankfurt (`diabetes_g.csv`)**, combined into a dataset containing 2,768 patient records.

The project includes data cleaning, missing-value imputation, feature engineering, standardization, class-imbalance handling with **SMOTE**, and an 80/20 train-test split.

Several models and ensemble techniques are evaluated, including **Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, MLP, Deep Neural Networks, Stacking, Voting, and Weighted Ensemble** methods.

The goal is to compare different machine learning approaches and develop a robust diabetes prediction system.

## 📊 Model Performance

Several machine learning and deep learning models were trained and evaluated for diabetes-risk prediction. The evaluation was performed using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.

### 🌲 Random Forest & Extra Trees

| Model           | Train Accuracy | Test Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| --------------- | -------------: | ------------: | ---------: | ---------: | ---------: | ---------: |
| Random Forest   |         98.33% |        93.86% |     92.43% |     89.53% |     90.96% |     98.73% |
| **Extra Trees** |       **100%** |    **99.28%** | **98.95%** | **98.95%** | **98.95%** | **99.98%** |

### 🚀 Gradient Boosting Models

| Model        | Train Accuracy | Test Accuracy |  Precision |     Recall |   F1-Score | ROC-AUC |
| ------------ | -------------: | ------------: | ---------: | ---------: | ---------: | ------: |
| XGBoost      |         99.77% |        98.01% |     95.45% |     98.95% |     97.17% |  99.89% |
| LightGBM     |         99.68% |        98.38% |     95.50% |   **100%** |     97.70% |  99.95% |
| **CatBoost** |       **100%** |    **99.28%** | **98.45%** | **99.48%** | **98.96%** |  99.97% |

### 🧠 Neural Networks

| Model                 | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------------------- | -------------: | ------------: | --------: | -----: | -------: | ------: |
| MLP                   |         87.44% |        84.12% |    77.84% | 75.39% |   76.60% |  91.95% |
| Deep Learning (Keras) |         97.52% |        94.04% |    89.50% | 93.72% |   91.56% |  98.68% |

### 🏆 Ensemble Methods

| Method                                      | Train Accuracy | Test Accuracy | Precision |     Recall | F1-Score | ROC-AUC |
| ------------------------------------------- | -------------: | ------------: | --------: | ---------: | -------: | ------: |
| Stacking (LightGBM + ExtraTrees + CatBoost) |           100% |        99.10% |    98.44% |     96.34% |   96.34% |  99.90% |
| Soft Voting                                 |         99.82% |        98.01% |    97.87% |     97.91% |   97.10% |  99.32% |
| Hard Voting                                 |         99.91% |        98.74% |    98.42% |     98.42% |   98.16% |       — |
| Weighted Ensemble                           |         99.95% |    **99.28%** |    98.45% | **99.48%** |   98.96% |  99.99% |

### 🥇 Best Results

Based on the test-set results:

* **Best Test Accuracy:** 99.28% — Extra Trees / CatBoost / Weighted Ensemble
* **Best Precision:** 98.95% — Extra Trees
* **Best Recall:** 100% — LightGBM
* **Best F1-Score:** 98.96% — CatBoost
* **Best ROC-AUC:** 99.99% — Weighted Ensemble


### ⚠️ Evaluation Note

The reported metrics should be interpreted in the context of the dataset, preprocessing pipeline, class distribution, and train/test methodology. High test performance does not necessarily imply equivalent performance on unseen real-world clinical populations.

