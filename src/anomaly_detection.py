"""
RetailIQ - Anomaly Detection Module
Implements Isolation Forest to detect unusual inventory movements,
sales spikes, sudden demand drops, and explains anomalous features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ANOMALY_MODEL_FILE = os.path.join(MODELS_DIR, "anomaly_model.pkl")

ANOMALY_FEATURES = [
    "Units Sold",
    "Sales_ZScore",
    "Discount",
    "Current Stock",
    "Stock_to_Reorder_Ratio",
    "Stock_Depletion"
]


def extract_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical deviation features across daily transactions:
    - Sales z-score relative to product's baseline demand
    - Inventory stock to reorder ratio
    - Daily stock depletion velocity
    """
    data = df.copy()
    data["Sales Date"] = pd.to_datetime(data["Sales Date"])
    data = data.sort_values(by=["Product ID", "Sales Date"]).reset_index(drop=True)

    # Calculate baseline demand per product
    prod_baselines = data.groupby("Product ID")["Units Sold"].agg(["mean", "std"]).reset_index()
    prod_baselines.columns = ["Product ID", "prod_mean_sales", "prod_std_sales"]
    prod_baselines["prod_std_sales"] = prod_baselines["prod_std_sales"].replace(0, 1.0).fillna(1.0)

    data = pd.merge(data, prod_baselines, on="Product ID", how="left")
    data["Sales_ZScore"] = (data["Units Sold"] - data["prod_mean_sales"]) / data["prod_std_sales"]

    # Stock dynamics
    data["Stock_to_Reorder_Ratio"] = (data["Current Stock"] / (data["Reorder Level"] + 1e-4)).clip(0, 10)

    # Previous day stock vs current day stock
    data["prev_stock"] = data.groupby("Product ID")["Current Stock"].shift(1).fillna(data["Current Stock"])
    data["Stock_Depletion"] = (data["prev_stock"] - data["Current Stock"]).clip(lower=0)

    return data


def train_anomaly_model(
    df: pd.DataFrame,
    contamination: float = 0.035,
    save_path: str = ANOMALY_MODEL_FILE,
    random_state: int = 42
) -> dict:
    """
    Fits an Isolation Forest model to detect sales and inventory anomalies.
    Computes anomaly scores, identifies feature drivers, and generates anomaly explanations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[ANOMALY DETECTION] Extracting anomaly detection features...")
    data_feat = extract_anomaly_features(df)

    X = data_feat[ANOMALY_FEATURES].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[ANOMALY DETECTION] Fitting Isolation Forest (contamination={contamination})...")
    iso_forest = IsolationForest(
        n_estimators=130,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    preds = iso_forest.fit_predict(X_scaled)
    # Decision function returns negative for anomalies, positive for inliers
    raw_scores = iso_forest.decision_function(X_scaled)
    # Convert to 0..1 intuitive anomaly score (higher = more anomalous)
    anomaly_scores = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-6)

    data_feat["Is_Anomaly"] = (preds == -1).astype(int)
    data_feat["Anomaly_Score"] = np.round(anomaly_scores, 4)

    # Generate Explainability / Anomaly Reason
    def explain_anomaly(row):
        if row["Is_Anomaly"] == 0:
            return "Normal Pattern"

        z_sales = row["Sales_ZScore"]
        units = row["Units Sold"]
        disc = row["Discount"]
        stock = row["Current Stock"]
        depletion = row["Stock_Depletion"]

        reasons = []
        if z_sales > 2.8:
            reasons.append(f"Unusual Demand Spike ({int(units)} units, +{z_sales:.1f}σ above baseline)")
        elif z_sales < -2.0 and row["prod_mean_sales"] > 5:
            reasons.append(f"Unexpected Demand Drop ({int(units)} units vs {row['prod_mean_sales']:.1f} avg)")

        if stock == 0 and units == 0:
            reasons.append("Critical Stockout Depletion (Zero Available Inventory)")
        elif depletion > (row["prod_mean_sales"] * 3.5):
            reasons.append(f"Rapid Stock Depletion ({int(depletion)} units drained in single cycle)")

        if disc >= 0.20 and z_sales < 0:
            reasons.append(f"Failed Promotion (High Discount {int(disc*100)}% with Sluggish Demand)")

        if not reasons:
            reasons.append(f"Multivariate deviation in sales ({int(units)}) and stock ({int(stock)})")

        return "; ".join(reasons)

    data_feat["Anomaly_Reason"] = data_feat.apply(explain_anomaly, axis=1)

    total_records = len(data_feat)
    anomaly_count = int(data_feat["Is_Anomaly"].sum())
    anomaly_pct = round((anomaly_count / total_records) * 100.0, 2)

    print(f"[ANOMALY DETECTION] Total records: {total_records:,}, Detected Anomalies: {anomaly_count:,} ({anomaly_pct}%)")

    # Filter anomaly dataset for reporting
    anomalies_df = data_feat[data_feat["Is_Anomaly"] == 1].sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)

    summary_stats = {
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "normal_count": total_records - anomaly_count,
        "anomaly_pct": anomaly_pct,
        "contamination": contamination,
        "features_used": ANOMALY_FEATURES
    }

    model_bundle = {
        "model": iso_forest,
        "scaler": scaler,
        "features": ANOMALY_FEATURES,
        "summary_stats": summary_stats,
        "anomalies_df": anomalies_df,
        "enriched_df": data_feat
    }

    joblib.dump(model_bundle, save_path)
    print(f"[ANOMALY DETECTION] Saved anomaly detection bundle to {save_path}")

    return model_bundle


def load_anomaly_model(model_path: str = ANOMALY_MODEL_FILE) -> dict:
    """Loads saved anomaly model bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Anomaly model not found at {model_path}. Train it first.")
    return joblib.load(model_path)


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import clean_retail_data

    raw = load_dataset()
    clean = clean_retail_data(raw)
    bundle = train_anomaly_model(clean)
    print("Top 5 Anomalies Detected:\n", bundle["anomalies_df"][["Sales Date", "Product ID", "Units Sold", "Anomaly_Score", "Anomaly_Reason"]].head())
