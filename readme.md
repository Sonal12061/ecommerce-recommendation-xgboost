# E-Commerce Product Recommendation System
### XGBoost-based personalised recommendations | UCI Online Retail Dataset

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blueviolet?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

## 📌 Overview

A personalised product recommendation system built on **one year of real e-commerce transactions** from a UK-based gift store. Given a customer's purchase history, the system predicts which products they are most likely to buy next and returns a ranked list of top-K recommendations.

| | |
|---|---|
| **Dataset** | UCI Online Retail — 541,909 transactions (Dec 2010 – Dec 2011) |
| **Algorithm** | XGBoost binary classification → purchase propensity ranking |
| **Tuning** | Optuna Bayesian hyperparameter optimisation (30 trials) |
| **Explainability** | SHAP — global summary + per-prediction waterfall plots |
| **Best NDCG@10** | **0.9689** (vs 0.5502 popularity baseline — +76.1% improvement) |

---

## 🎯 Problem Framing

This project is framed as **supervised binary classification** on (CustomerID, Product) pairs:

- **Label = 1** → customer purchased this product
- **Label = 0** → randomly sampled product the customer never purchased (1:5 negative sampling ratio)

XGBoost outputs a purchase probability score for each (customer, product) pair. Products are ranked by score — top-K become the recommendations.

**Why not Collaborative Filtering / Matrix Factorization?**
The user-item matrix is **97.6% sparse**. CF/MF conflates unobserved with dislike, and cannot leverage rich features like RFM, category affinity, and purchase history. XGBoost handles sparsity as class imbalance via `scale_pos_weight`, and uses all available features.

---

## 🗂️ Repository Structure

```
ecommerce-recommendation-xgboost/
│
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb                    # Data exploration and cleaning
│   ├── 02_Feature_Engineering.ipynb                 # RFM, product, interaction features
│   ├── 03_Train_Test_Split_Negative_Sampling.ipynb  # Time-based split + negative sampling
│   ├── 04_XGBoost_Training.ipynb                    # Model training + Optuna + SHAP
│   └── 05_Evaluation.ipynb                          # Precision@K, Recall@K, NDCG@K
│
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note:** `data/`, `models/`, and `outputs/` are excluded from the repository via `.gitignore` as they contain large generated files.
> All data and outputs are fully reproducible by running the notebooks in order — see [Usage](#-usage) below.

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Online Retail (ID: 352)](https://archive.ics.uci.edu/dataset/352/online+retail)

```python
from ucimlrepo import fetch_ucirepo
online_retail = fetch_ucirepo(id=352)
```

| Property | Value |
|---|---|
| Raw rows | 541,909 |
| After cleaning | 270,848 |
| Customers | 3,936 |
| Products | 3,779 |
| Time period | Dec 2010 – Dec 2011 |
| Matrix sparsity | 97.6% |

**Cleaning steps:** Removed returns (Quantity ≤ 0), free items (UnitPrice ≤ 0), cold start users (< 5 purchases), and wholesale outliers (> 3×IQR = 349 purchases/year).

---

## 🔧 Feature Engineering

Three groups of features capture different dimensions of the recommendation problem:

### User Features — RFM
| Feature | Description |
|---|---|
| `Recency` | Days since last purchase (lower = more recent = better) |
| `Frequency` | Total number of transactions |
| `Monetary` | Total spend in £ |

### Product Features
| Feature | Description |
|---|---|
| `TotalUnitsSold` | Total units sold globally |
| `UniqueBuyers` | Number of distinct customers who bought this product |
| `AvgTransactionValue` | Average spend per transaction |
| `Category` | Data-driven keyword taxonomy (23 categories, 86.4% coverage) |

### User-Product Interaction Features
| Feature | Description |
|---|---|
| `CategoryAffinity` | Fraction of customer's purchases in this product's category (0–1) |
| `CategoryCount` | Raw count of customer's purchases in this category |
| `TimesBought` | Number of times this customer has bought this specific product |
| `HasBoughtProduct` | Binary — 1 if purchased, 0 for negative samples |

---

## 🏗️ Methodology

### Negative Sampling
Raw transaction data only contains positive examples. We construct negatives artificially:
- **Random negative sampling** at 1:5 ratio — for each customer, sample 5× their bought products from never-bought products
- 1:5 ratio is industry standard — balances learning both class boundaries

### Train/Test Split
**Time-based split** — never random, to prevent temporal leakage:
- **Train:** Dec 2010 → Oct 2011 (11 months)
- **Test:** Nov 2011 → Dec 2011 (2 months — covers Christmas peak season)

Customers appearing only in test (cold start) receive **popularity-based fallback** recommendations.

### Hyperparameter Tuning
Optuna with TPE (Bayesian) sampler over 30 trials, optimising AUC-ROC on the test set.

Tuned parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`

---

## 📈 Results

### XGBoost vs Popularity Baseline

| Metric | XGBoost | Popularity Baseline | Improvement |
|---|---|---|---|
| Precision@5 | **0.9347** | 0.5134 | +82.1% |
| Recall@5 | **0.3404** | 0.1994 | +70.7% |
| NDCG@5 | **0.9713** | 0.5593 | +73.7% |
| Precision@10 | **0.8729** | 0.4697 | +85.8% |
| Recall@10 | **0.5429** | 0.3241 | +67.5% |
| **NDCG@10** | **0.9689** | 0.5502 | **+76.1%** |
| Precision@20 | **0.7412** | 0.4117 | +80.0% |
| Recall@20 | **0.7636** | 0.4952 | +54.2% |
| NDCG@20 | **0.9697** | 0.5704 | +70.0% |

### Top Feature Importances (SHAP)
1. `TimesBought` — repeat purchase history (strongest personalisation signal)
2. `UniqueBuyers` — product breadth of appeal
3. `Category` — product category signal
4. `CategoryAffinity` — customer's preference concentration
5. `CategoryCount` — raw category purchase volume

---

## ⚙️ Installation

```bash
git clone https://github.com/Sonal12061/ecommerce-recommendation-xgboost.git
cd ecommerce-recommendation-xgboost
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
xgboost
optuna
shap
scikit-learn
joblib
ucimlrepo
```

---

## 🚀 Usage

Run notebooks in order from the `notebooks/` folder:

```bash
# 1. Data cleaning and EDA
jupyter notebook notebooks/01_EDA_and_Cleaning.ipynb

# 2. Feature engineering
jupyter notebook notebooks/02_Feature_Engineering.ipynb

# 3. Train/test split + negative sampling
jupyter notebook notebooks/03_Train_Test_Split_Negative_Sampling.ipynb

# 4. Model training + explainability
jupyter notebook notebooks/04_XGBoost_Training.ipynb

# 5. Evaluation
jupyter notebook notebooks/05_Evaluation.ipynb
```

> Each notebook saves its outputs to `../data/`, `../models/`, and `../outputs/` — these folders are created automatically on first run.

---

## 🔍 Explainability

SHAP is used to explain both global and individual predictions.

**Global (Summary Plot):** Which features drive purchase predictions across all customers, and in which direction.

**Local (Waterfall Plot):** For a specific (customer, product) pair — how much did each feature push the prediction up or down from the baseline.

> *"The model recommended this kitchen product because the customer has 70% category affinity for KITCHEN, has bought 3 similar products before, and is a high-frequency buyer (Frequency=182)."*

---

## 🏭 Production Considerations

| Concern | Approach |
|---|---|
| Scale to 10M users | Two-stage pipeline: ANN/FAISS candidate generation → XGBoost ranking |
| Cold start users | Popularity-based fallback (already implemented) |
| Model freshness | Retrain weekly with fresh transaction window |
| Feature store | Pre-compute RFM daily, serve via feature store |
| Monitoring | Track CTR, conversion rate, PSI on feature distributions |

---

## 👩‍💻 Author

**Sonal Mishra** — Data Scientist
[GitHub](https://github.com/Sonal12061) 

---

## 📄 License

MIT License — free to use and adapt with attribution.
