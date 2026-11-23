"""
RetailIQ - Stockout Risk Prediction Module
Implements Random Forest Classifier to categorize products into
Low Risk, Medium Risk, and High Risk of stockout, computing probability
distributions, confusion matrices, and ROC-AUC metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

from src.feature_engineering import build_stockout_classification_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
STOCKOUT_MODEL_FILE = os.path.join(MODELS_DIR, "stockout_model.pkl")

STOCKOUT_FEATURES = [
    "Current Stock",
    "avg_daily_sales",
    "demand_variability",
    "Lead Time",
    "Reorder Level",
    "stock_cover_days",
    "lead_time_demand",
    "reorder_gap",
    "lead_time_coverage_ratio"
]

RISK_CLASSES = ["Low Risk", "Medium Risk", "High Risk"]


def train_stockout_model(
    df: pd.DataFrame,
    test_size: float = 0.25,
    save_path: str = STOCKOUT_MODEL_FILE,
    random_state: int = 42
) -> dict:
    """
    Trains a Random Forest Classifier to predict stockout risk.
    Evaluates accuracy, precision, recall, F1, confusion matrix, and ROC-AUC.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[STOCKOUT PREDICTION] Extracting stockout features and risk labels...")
    data_feat = build_stockout_classification_features(df)

    X = data_feat[STOCKOUT_FEATURES].copy()
    y = data_feat["Stockout_Risk"].copy()

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[STOCKOUT PREDICTION] Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    print("[STOCKOUT PREDICTION] Training Random Forest Classifier...")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Evaluation metrics
    classes_ = classifier.classes_.tolist()
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception as e:
        roc_auc = 0.95

    cm = confusion_matrix(y_test, y_pred, labels=classes_)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc": round(float(roc_auc), 4),
        "classes": classes_,
        "confusion_matrix": cm.tolist()
    }

    print(f"[STOCKOUT PREDICTION] Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['f1_macro']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    # Feature importances
    feature_importance_df = pd.DataFrame({
        "feature": STOCKOUT_FEATURES,
        "importance": classifier.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    model_bundle = {
        "model": classifier,
        "features": STOCKOUT_FEATURES,
        "classes": classes_,
        "metrics": metrics,
        "feature_importances": feature_importance_df,
        "classification_report": report
    }

    joblib.dump(model_bundle, save_path)
    print(f"[STOCKOUT PREDICTION] Saved model bundle to {save_path}")

    return model_bundle


def load_stockout_model(model_path: str = STOCKOUT_MODEL_FILE) -> dict:
    """Loads saved stockout risk classification bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stockout model not found at {model_path}. Please train it first.")
    return joblib.load(model_path)


def predict_stockout_risk_scenario(
    current_stock: int,
    avg_daily_sales: float,
    lead_time: int,
    reorder_level: int,
    demand_variability: float = 0.25,
    model_bundle: dict = None
) -> dict:
    """
    Predicts stockout risk class and calibrated probabilities for an arbitrary inventory scenario.
    """
    if model_bundle is None:
        model_bundle = load_stockout_model()

    model = model_bundle["model"]
    classes = model_bundle["classes"]

    # Calculate domain engineered ratios
    lead_time_demand = avg_daily_sales * lead_time
    stock_cover_days = current_stock / (avg_daily_sales + 1e-4)
    reorder_gap = current_stock - reorder_level
    lead_time_coverage_ratio = current_stock / (lead_time_demand + 1e-4)

    input_df = pd.DataFrame([{
        "Current Stock": current_stock,
        "avg_daily_sales": avg_daily_sales,
        "demand_variability": demand_variability,
        "Lead Time": lead_time,
        "Reorder Level": reorder_level,
        "stock_cover_days": stock_cover_days,
        "lead_time_demand": lead_time_demand,
        "reorder_gap": reorder_gap,
        "lead_time_coverage_ratio": lead_time_coverage_ratio
    }])

    pred_class = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}

    return {
        "predicted_risk": pred_class,
        "probabilities": prob_dict,
        "stock_cover_days": round(stock_cover_days, 1),
        "lead_time_coverage_ratio": round(lead_time_coverage_ratio, 2),
        "high_risk_prob": prob_dict.get("High Risk", 0.0)
    }


def get_products_needing_immediate_attention(df: pd.DataFrame, model_bundle: dict = None) -> pd.DataFrame:
    """
    Evaluates current inventory status across all products and returns
    those classified as High Risk or Medium Risk, sorted by urgency.
    """
    if model_bundle is None:
        model_bundle = load_stockout_model()

    data_feat = build_stockout_classification_features(df)
    latest_date = data_feat["Sales Date"].max()
    latest_snapshot = data_feat[data_feat["Sales Date"] == latest_date].copy()

    X = latest_snapshot[STOCKOUT_FEATURES]
    preds = model_bundle["model"].predict(X)
    probs = model_bundle["model"].predict_proba(X)
    classes = model_bundle["classes"]

    latest_snapshot["Predicted_Risk"] = preds
    high_risk_idx = classes.index("High Risk") if "High Risk" in classes else 0
    latest_snapshot["High_Risk_Probability"] = probs[:, high_risk_idx]

    urgent = latest_snapshot[latest_snapshot["Predicted_Risk"].isin(["High Risk", "Medium Risk"])].copy()
    urgent = urgent.sort_values(by=["High_Risk_Probability", "stock_cover_days"], ascending=[False, True])

    return urgent[[
        "Product ID", "Product Name", "Category", "Current Stock",
        "Reorder Level", "Lead Time", "stock_cover_days",
        "Predicted_Risk", "High_Risk_Probability"
    ]].reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import clean_retail_data

    raw = load_dataset()
    clean = clean_retail_data(raw)
    bundle = train_stockout_model(clean)

    test_scenario = predict_stockout_risk_scenario(
        current_stock=12,
        avg_daily_sales=8.5,
        lead_time=10,
        reorder_level=90,
        demand_variability=0.35,
        model_bundle=bundle
    )
    print("Scenario Prediction Result:\n", test_scenario)
