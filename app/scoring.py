"""Credit risk scoring utilities."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "credit_risk_model.pkl"
DATA_PATH = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
REPORTS_DIR = ROOT / "reports"

MODEL_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "TOTAL_BILL", "TOTAL_PAYMENT", "UTILIZATION_RATIO", "AVG_DELAY",
]

RAW_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

MODEL_METRICS = {
    "XGBoost": 0.7789,
    "Random Forest": 0.7778,
    "Logistic Regression + SMOTE": 0.7117,
    "Logistic Regression": 0.7084,
}

__all__ = [
    "MODEL_FEATURES",
    "MODEL_METRICS",
    "RAW_COLUMNS",
    "REPORTS_DIR",
    "assign_decision",
    "assign_risk_category",
    "assign_risk_category_custom",
    "build_feature_row",
    "engineer_features",
    "load_model",
    "load_sample_data",
    "predict",
    "predict_batch",
    "predict_with_thresholds",
    "probability_to_score",
    "risk_color",
    "sensitivity_analysis",
]


def load_model():
    return joblib.load(MODEL_PATH)


def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"default.payment.next.month": "default"})
    df.drop("ID", axis=1, inplace=True)
    return engineer_features(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features matching the trained model."""
    result = df.copy()
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    payment_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_cols = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]

    result["TOTAL_BILL"] = result[bill_cols].sum(axis=1)
    result["TOTAL_PAYMENT"] = result[payment_cols].sum(axis=1)
    result["UTILIZATION_RATIO"] = result["TOTAL_BILL"] / (result["LIMIT_BAL"] + 1)
    result["AVG_DELAY"] = result[pay_cols].mean(axis=1)
    return result


def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {col: inputs[col] for col in RAW_COLUMNS}
    return engineer_features(pd.DataFrame([row]))


def probability_to_score(probability: float) -> int:
    return int(850 - probability * 550)


def assign_risk_category(score: int) -> str:
    if score >= 800:
        return "Very Low Risk"
    if score >= 700:
        return "Low Risk"
    if score >= 600:
        return "Medium Risk"
    return "High Risk"


def assign_decision(score: int, approve_min: int = 700, review_min: int = 600) -> str:
    if score >= approve_min:
        return "Auto-Approve"
    if score >= review_min:
        return "Manual Review"
    return "Auto-Decline"


def assign_risk_category_custom(score: int, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {"very_low": 800, "low": 700, "medium": 600}
    if score >= thresholds["very_low"]:
        return "Very Low Risk"
    if score >= thresholds["low"]:
        return "Low Risk"
    if score >= thresholds["medium"]:
        return "Medium Risk"
    return "High Risk"


def risk_color(category: str) -> str:
    return {
        "Very Low Risk": "#16a34a",
        "Low Risk": "#22c55e",
        "Medium Risk": "#f59e0b",
        "High Risk": "#ef4444",
    }.get(category, "#6b7280")


def predict(model, features: pd.DataFrame) -> dict:
    x = features[MODEL_FEATURES]
    probability = float(model.predict_proba(x)[:, 1][0])
    score = probability_to_score(probability)
    return {
        "default_probability": probability,
        "credit_score": score,
        "risk_category": assign_risk_category(score),
        "decision": assign_decision(score),
    }


def predict_with_thresholds(
    model,
    features: pd.DataFrame,
    approve_min: int = 700,
    review_min: int = 600,
) -> dict:
    result = predict(model, features)
    score = result["credit_score"]
    result["risk_category"] = assign_risk_category_custom(
        score,
        {"very_low": 800, "low": approve_min, "medium": review_min},
    )
    result["decision"] = assign_decision(score, approve_min, review_min)
    return result


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    features = engineer_features(df)
    x = features[MODEL_FEATURES]
    probabilities = model.predict_proba(x)[:, 1]
    scores = [probability_to_score(p) for p in probabilities]
    return pd.DataFrame({
        "default_probability": probabilities,
        "credit_score": scores,
        "risk_category": [assign_risk_category(s) for s in scores],
        "decision": [assign_decision(s) for s in scores],
    })


def sensitivity_analysis(
    model,
    base_inputs: dict,
    feature: str,
    values: list,
    approve_min: int = 700,
    review_min: int = 600,
) -> pd.DataFrame:
    rows = []
    for value in values:
        inputs = dict(base_inputs)
        inputs[feature] = value
        features = build_feature_row(inputs)
        result = predict_with_thresholds(model, features, approve_min, review_min)
        rows.append({
            feature: value,
            "credit_score": result["credit_score"],
            "default_probability": result["default_probability"],
            "decision": result["decision"],
        })
    return pd.DataFrame(rows)
