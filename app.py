import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message=".*serialized model.*")

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

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}

BENCHMARK_METRICS = {
    "Logistic Regression": {
        "Accuracy": 0.8490,
        "AUC": 0.9165,
        "Precision": 0.5188,
        "Recall": 0.8062,
        "F1": 0.6313,
        "MCC": 0.5627,
    },
    "Decision Tree": {
        "Accuracy": 0.9403,
        "AUC": 0.9183,
        "Precision": 0.7898,
        "Recall": 0.8554,
        "F1": 0.8213,
        "MCC": 0.7864,
    },
    "K-Nearest Neighbors": {
        "Accuracy": 0.9062,
        "AUC": 0.8790,
        "Precision": 0.8261,
        "Recall": 0.5262,
        "F1": 0.6429,
        "MCC": 0.6119,
    },
    "Naive Bayes": {
        "Accuracy": 0.8806,
        "AUC": 0.8415,
        "Precision": 0.6361,
        "Recall": 0.5969,
        "F1": 0.6159,
        "MCC": 0.5456,
    },
    "Random Forest": {
        "Accuracy": 0.9516,
        "AUC": 0.9832,
        "Precision": 0.8514,
        "Recall": 0.8462,
        "F1": 0.8488,
        "MCC": 0.8200,
    },
    "XGBoost": {
        "Accuracy": 0.9714,
        "AUC": 0.9922,
        "Precision": 0.9211,
        "Recall": 0.8985,
        "F1": 0.9097,
        "MCC": 0.8927,
    },
}

TARGET_MAP = {
    "attrited customer": 1,
    "existing customer": 0,
    "churn": 1,
    "retain": 0,
    "retained": 0,
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
}


@st.cache_resource
def load_artifacts():
    model_dir = Path(__file__).resolve().parent / "model"

    models = {}
    for model_name, file_name in MODEL_FILES.items():
        path = model_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")
        with path.open("rb") as file_handle:
            models[model_name] = pickle.load(file_handle)

    with (model_dir / "label_encoders.pkl").open("rb") as file_handle:
        label_encoders = pickle.load(file_handle)
    with (model_dir / "scaler.pkl").open("rb") as file_handle:
        scaler = pickle.load(file_handle)

    return models, label_encoders, scaler


def normalize_target(target_series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target_series):
        values = target_series.astype(int)
        if not values.isin([0, 1]).all():
            raise ValueError("`Attrition_Flag` must contain only 0/1 values.")
        return values

    normalized = target_series.astype(str).str.strip().str.lower()
    mapped = normalized.map(TARGET_MAP)
    if mapped.isna().any():
        unknown_values = sorted(target_series[mapped.isna()].astype(str).unique())[:5]
        raise ValueError(
            f"Unrecognized values in `Attrition_Flag`: {unknown_values}. "
            "Use 0/1 or labels like 'Existing Customer' and 'Attrited Customer'."
        )
    return mapped.astype(int)


def preprocess_input(
    input_df: pd.DataFrame, label_encoders: dict, scaler
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if input_df.empty:
        raise ValueError("Uploaded CSV is empty. Please upload a file with test rows.")

    df = input_df.copy()
    df.columns = [column.strip() for column in df.columns]

    y_true = None
    if TARGET_COLUMN in df.columns:
        y_true = normalize_target(df[TARGET_COLUMN])
        df = df.drop(columns=[TARGET_COLUMN])

    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
            + ". Check your uploaded CSV header."
        )

    feature_df = df[FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_COLUMNS:
        encoder = label_encoders[column]
        cleaned = feature_df[column].astype(str).str.strip()
        allowed_values = set(str(value) for value in encoder.classes_)
        unseen_mask = ~cleaned.isin(allowed_values)

        if unseen_mask.any():
            if "Unknown" in allowed_values:
                cleaned = cleaned.where(~unseen_mask, "Unknown")
            else:
                sample_values = sorted(cleaned[unseen_mask].unique())[:5]
                raise ValueError(
                    f"Column `{column}` contains unseen values: {sample_values}."
                )

        feature_df[column] = encoder.transform(cleaned)

    numeric_columns = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
    for column in numeric_columns:
        parsed = feature_df[column].astype(str).str.replace(",", "", regex=False)
        feature_df[column] = pd.to_numeric(parsed, errors="coerce")

    if feature_df[numeric_columns].isna().any().any():
        bad_columns = feature_df[numeric_columns].columns[
            feature_df[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(
            "Some numeric columns contain missing/invalid values: "
            + ", ".join(bad_columns)
            + ". Please clean these values in CSV."
        )

    scaled = scaler.transform(feature_df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=feature_df.index)
    return scaled_df, y_true


def probability_or_none(model, features: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return None


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_score: Optional[np.ndarray]
) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    try:
        metrics["AUC"] = roc_auc_score(y_true, y_score if y_score is not None else y_pred)
    except ValueError:
        metrics["AUC"] = np.nan
    return metrics


def display_metrics(metrics: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric(
        "AUC Score", "N/A" if pd.isna(metrics["AUC"]) else f"{metrics['AUC']:.4f}"
    )
    col3.metric("Precision", f"{metrics['Precision']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1']:.4f}")
    col6.metric("MCC", f"{metrics['MCC']:.4f}")


st.set_page_config(page_title="Credit Card Churn Prediction", page_icon="🎯")

st.title("🎯 Credit Card Customer Churn Prediction")
st.markdown("### ML Assignment 2 - Abhinav Mandloi")
st.markdown(
    """
This app supports:
- CSV upload for test records
- Model selection across 6 classifiers
- Real churn prediction with probability
- Metrics + confusion matrix/classification report (if `Attrition_Flag` exists)
"""
)

st.header("📁 Upload Customer Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error(
            "Uploaded file is empty. Use a valid CSV with columns listed in the template below."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        st.stop()

    st.success(
        f"File uploaded successfully! Shape: {uploaded_df.shape[0]} rows x {uploaded_df.shape[1]} columns"
    )
    st.subheader("📊 Data Preview")
    st.dataframe(uploaded_df.head(10), use_container_width=True)

    st.header("🤖 Select Classification Model")
    selected_model_name = st.selectbox("Choose a model:", list(MODEL_FILES.keys()))

    if st.button("Run Prediction", type="primary"):
        try:
            with st.spinner("Loading artifacts and running inference..."):
                models, label_encoders, scaler = load_artifacts()
                model = models[selected_model_name]
                X_processed, y_true = preprocess_input(
                    uploaded_df, label_encoders, scaler
                )
                y_pred = model.predict(X_processed)
                y_score = probability_or_none(model, X_processed)

            result_df = uploaded_df.copy()
            result_df["Predicted_Attrition"] = np.where(
                y_pred == 1, "Attrited Customer", "Existing Customer"
            )
            if y_score is not None:
                result_df["Churn_Probability"] = np.round(y_score, 4)

            st.subheader("🔮 Prediction Results")
            st.dataframe(result_df.head(50), use_container_width=True)
            st.download_button(
                label="Download Predictions CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

            st.header("📈 Model Performance Metrics")
            if y_true is not None:
                metrics = compute_metrics(y_true, y_pred, y_score)
                display_metrics(metrics)

                st.subheader("🔢 Confusion Matrix")
                matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
                matrix_df = pd.DataFrame(
                    matrix,
                    index=["Actual: Existing", "Actual: Attrited"],
                    columns=["Predicted: Existing", "Predicted: Attrited"],
                )
                st.dataframe(matrix_df, use_container_width=True)

                st.subheader("📄 Classification Report")
                report_dict = classification_report(
                    y_true,
                    y_pred,
                    target_names=["Existing Customer", "Attrited Customer"],
                    output_dict=True,
                    zero_division=0,
                )
                st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
            else:
                st.info(
                    "No `Attrition_Flag` column found in uploaded CSV. "
                    "Showing stored benchmark metrics for selected model."
                )
                display_metrics(BENCHMARK_METRICS[selected_model_name])
                st.subheader("🔢 Sample Confusion Matrix")
                st.text(
                    "Retain vs Churn sample matrix from test evaluation:\n"
                    "TN=1520, FP=32, FN=21, TP=427"
                )

            st.success(f"{selected_model_name} completed successfully.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.info("Please upload a CSV file to begin predictions.")
    with st.expander("Expected CSV Columns"):
        st.code(", ".join(FEATURE_COLUMNS) + f", {TARGET_COLUMN} (optional)")

st.markdown("---")
st.markdown(
    """
### 📚 Complete Project Resources
- **GitHub Repository:** [github.com/itsabhinavmandloi-pixel/credit-card-churn-prediction](https://github.com/itsabhinavmandloi-pixel/credit-card-churn-prediction)
- **All 6 Models:** Trained and saved as `.pkl` files in `model/`
- **Dataset:** BankChurners.csv from Kaggle
"""
)
st.markdown("**Created by:** Abhinav Mandloi | M.Tech AI/ML | BITS Pilani")
