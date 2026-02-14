"""Train, evaluate, and persist all six churn-classification models.

Usage:
    python model/train_models.py --data /path/to/BankChurners.csv --outdir model
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]
CATEGORICAL_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
TARGET_COLUMN = "Attrition_Flag"

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data", required=True, help="Path to BankChurners CSV file")
    parser.add_argument("--outdir", default="model", help="Output directory for artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_csv(path)

    # Keep only required feature columns and target; ignore helper columns in raw dataset.
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    data = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    label_encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str).str.strip())
        label_encoders[col] = encoder

    y = data[TARGET_COLUMN].astype(str).str.strip().map(
        {"Existing Customer": 0, "Attrited Customer": 1}
    )
    if y.isna().any():
        bad_values = sorted(data[TARGET_COLUMN][y.isna()].astype(str).unique())
        raise ValueError(f"Unknown target labels found: {bad_values}")

    X = data[FEATURE_COLUMNS].copy()
    return X, y.astype(int), label_encoders


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    imbalance_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=random_state, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state, class_weight="balanced"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=imbalance_ratio,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    metric_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)
        metric_rows.append({"Model": model_name, **metrics})
        trained_models[model_name] = model

    metrics_df = pd.DataFrame(metric_rows).sort_values(by="AUC", ascending=False)
    return trained_models, metrics_df


def save_artifacts(
    outdir: Path,
    models: dict,
    scaler: StandardScaler,
    label_encoders: dict,
    metrics_df: pd.DataFrame,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        file_name = MODEL_FILE_MAP[model_name]
        with (outdir / file_name).open("wb") as f:
            pickle.dump(model, f)

    with (outdir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    with (outdir / "label_encoders.pkl").open("wb") as f:
        pickle.dump(label_encoders, f)

    metrics_df.to_csv(outdir / "model_metrics.csv", index=False)
    with (outdir / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.data)
    outdir = Path(args.outdir)

    X, y, label_encoders = load_and_prepare_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)

    models, metrics_df = train_and_evaluate(
        X_train_scaled_df, y_train, X_test_scaled, y_test, args.random_state
    )

    save_artifacts(outdir, models, scaler, label_encoders, metrics_df)
    print("Training complete. Saved artifacts in:", outdir.resolve())
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
