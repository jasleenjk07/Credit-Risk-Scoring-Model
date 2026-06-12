# Credit Risk Scoring Model

A machine learning system that predicts the probability of credit card default and converts it into an actionable credit score (300–900 scale). Banks, NBFCs, and fintechs use models like this to automate lending decisions — approving, rejecting, or flagging applications for manual review.

## Overview

This project implements an end-to-end credit risk scoring pipeline using the [UCI Credit Card Default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) (30,000 clients, 24 features). The workflow covers data preprocessing, feature engineering, model training, score calibration, risk banding, and model interpretability via SHAP.

**Target variable:** `default` — binary flag (1 = default next month, 0 = no default)

## Pipeline Architecture

![Credit Risk Scoring Pipeline Architecture](./reports/pipeline_architecture.png)

```
Data Ingestion → Preprocessing → Feature Engineering → Model Training
       → Validation → Score Calibration & Banding → Decision Engine
```

| Stage | What it does |
|---|---|
| **Data Ingestion** | Load UCI Credit Card dataset from `data/raw/` |
| **Preprocessing** | Handle missing values, scale numeric features with `StandardScaler` |
| **Feature Engineering** | Derive payment ratios, utilization, delinquency signals |
| **Model Training** | Train Logistic Regression, Random Forest, and XGBoost classifiers |
| **Validation** | Evaluate with ROC-AUC, classification report, confusion matrix |
| **Score Calibration** | Map default probability → credit score (300–900) |
| **Risk Banding** | Assign risk categories for decision routing |
| **Interpretability** | Feature importance and SHAP summary plots |

## Model Results

Four models were trained and compared on an 80/20 train-test split:

| Model | ROC-AUC |
|---|---|
| **XGBoost** | **0.7789** |
| Random Forest | 0.7778 |
| Logistic Regression + SMOTE | 0.7117 |
| Logistic Regression | 0.7084 |

XGBoost achieved the best discriminatory power and was selected as the production model. Class imbalance was addressed using SMOTE for the logistic regression variant.

## Feature Engineering

Engineered features from raw bureau and repayment data:

| Feature | Description |
|---|---|
| `TOTAL_BILL` | Sum of bill amounts across 6 months |
| `TOTAL_PAYMENT` | Sum of payment amounts across 6 months |
| `PAYMENT_RATIO` | Ratio of total payments to total bills |
| `CREDIT_AVAILABLE` | Remaining credit headroom (`LIMIT_BAL - TOTAL_BILL`) |
| `UTILIZATION_RATIO` | Credit utilization (`TOTAL_BILL / LIMIT_BAL`) |
| `PAYMENT_DEFICIT` | Unpaid balance (`TOTAL_BILL - TOTAL_PAYMENT`) |
| `AVG_DELAY` | Average repayment delay across 6 months |

## Score Calibration & Risk Bands

Default probabilities from XGBoost are converted to a credit score:

```
Credit Score = 850 − (default_probability × 550)
```

| Score Range | Risk Category | Decision |
|---|---|---|
| ≥ 800 | Very Low Risk | Auto-approve |
| 700 – 799 | Low Risk | Auto-approve |
| 600 – 699 | Medium Risk | Manual review |
| < 600 | High Risk | Auto-decline |

## Visualizations & Model Interpretability

### Feature Importance (XGBoost)

`PAY_0` (most recent repayment status) is the dominant predictor, followed by `AVG_DELAY` and historical payment behavior (`PAY_2`, `PAY_3`).

![XGBoost Feature Importance](./reports/feature_importance.png)

### SHAP Summary

SHAP values show how each feature pushes predictions toward default or non-default. Red (high feature value) and blue (low feature value) indicate direction and magnitude of impact.

![SHAP Summary Plot](./reports/shap_summary.png)

### Credit Score Distribution

Calibrated scores are left-skewed — most applicants cluster in the 750–850 range, with a smaller high-risk tail below 600.

![Credit Score Distribution](./reports/credit_score_distribution.png)

## Project Structure

```
Credit-Risk-Scoring-Model/
├── data/
│   └── raw/
│       └── UCI_Credit_Card.csv      # Source dataset
├── notebooks/
│   └── Credit_Risk_Model.ipynb      # End-to-end training notebook
├── models/
│   ├── credit_risk_model.pkl        # Trained XGBoost model
│   └── scaler.pkl                   # Fitted StandardScaler
├── reports/
│   ├── pipeline_architecture.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── credit_score_distribution.png
├── app/
│   ├── streamlit_app.py             # Streamlit dashboard
│   └── scoring.py                   # Scoring utilities
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone and set up environment

```bash
git clone <repository-url>
cd Credit-Risk-Scoring-Model

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the training notebook

```bash
jupyter notebook notebooks/Credit_Risk_Model.ipynb
```

The notebook will:
- Load and preprocess the dataset
- Engineer features and train all models
- Calibrate scores and assign risk bands
- Save model artifacts to `models/` and plots to `reports/`

### 3. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes:
- **Score Applicant** — interactive form with live credit score and decision
- **Batch Scoring** — upload CSV or score sample records in bulk
- **Model Insights** — ROC-AUC comparison and interpretability plots
- **Portfolio Analytics** — risk band and default rate analysis

### 4. Load the trained model

```python
import joblib

model = joblib.load("models/credit_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Predict default probability
prob = model.predict_proba(scaled_features)[:, 1]
credit_score = int(850 - prob * 550)
```

## Tech Stack

| Library | Purpose |
|---|---|
| pandas / numpy | Data manipulation |
| scikit-learn | Preprocessing, Logistic Regression, Random Forest |
| XGBoost | Gradient boosting classifier |
| imbalanced-learn | SMOTE for class imbalance |
| SHAP | Model explainability |
| matplotlib / seaborn | Visualization |
| Jupyter | Interactive development |

## Dataset

**Source:** [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

- **Records:** 30,000
- **Features:** 23 input variables + 1 target
- **Default rate:** ~22%
- **Key raw features:** `LIMIT_BAL`, `PAY_0`–`PAY_6` (repayment status), `BILL_AMT1`–`BILL_AMT6`, `PAY_AMT1`–`PAY_AMT6`, demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`)

## Future Work

- [ ] FastAPI REST endpoint for real-time scoring
- [x] Streamlit dashboard for interactive what-if analysis
- [ ] KS statistic, Gini coefficient, and PSI monitoring
- [ ] Out-of-time validation split
- [ ] Scheduled retraining on score drift alerts

## License

This project is for educational and portfolio purposes. The UCI dataset is publicly available for research use.
# Credit-Risk-Scoring-Model
# Credit-Risk-Scoring-Model
