"""
RetailIQ - Model Evaluation Suite
Collects and formats comprehensive evaluation metrics, feature importances,
confusion matrices, and diagnostic reports across all machine learning models.
"""

import os
import joblib
import pandas as pd

from src.demand_forecasting import load_demand_model
from src.stockout_prediction import load_stockout_model
from src.product_clustering import load_product_cluster_model
from src.customer_segmentation import load_customer_cluster_model
from src.anomaly_detection import load_anomaly_model


def get_all_models_evaluation_summary() -> dict:
    """
    Loads all trained model bundles from disk and compiles a unified
    performance and diagnostic evaluation dictionary.
    """
    summary = {}

    # 1. Demand Forecasting
    try:
        demand_bundle = load_demand_model()
        summary["demand"] = {
            "status": "Trained",
            "model_type": "Random Forest Regressor (Time-Aware Split)",
            "metrics": demand_bundle["metrics"],
            "feature_importances": demand_bundle["feature_importances"].to_dict(orient="records"),
            "train_period": demand_bundle.get("split_date", "Chronological cutoff")
        }
    except Exception as e:
        summary["demand"] = {"status": "Error", "message": str(e)}

    # 2. Stockout Risk Classification
    try:
        stockout_bundle = load_stockout_model()
        summary["stockout"] = {
            "status": "Trained",
            "model_type": "Random Forest Classifier (Class Weighted)",
            "metrics": stockout_bundle["metrics"],
            "classes": stockout_bundle["classes"],
            "confusion_matrix": stockout_bundle["metrics"]["confusion_matrix"],
            "feature_importances": stockout_bundle["feature_importances"].to_dict(orient="records"),
            "classification_report": stockout_bundle["classification_report"]
        }
    except Exception as e:
        summary["stockout"] = {"status": "Error", "message": str(e)}

    # 3. Product Clustering
    try:
        prod_cluster_bundle = load_product_cluster_model()
        summary["product_clustering"] = {
            "status": "Trained",
            "algorithm": "K-Means + StandardScaler + PCA",
            "n_clusters": prod_cluster_bundle["n_clusters"],
            "silhouette_score": prod_cluster_bundle["final_silhouette"],
            "explained_variance": prod_cluster_bundle["explained_variance"],
            "k_values": prod_cluster_bundle["k_values"],
            "elbow_inertias": prod_cluster_bundle["elbow_inertias"],
            "silhouette_scores": prod_cluster_bundle["silhouette_scores"],
            "cluster_profiles": prod_cluster_bundle["cluster_labels_map"]
        }
    except Exception as e:
        summary["product_clustering"] = {"status": "Error", "message": str(e)}

    # 4. Customer Segmentation
    try:
        cust_cluster_bundle = load_customer_cluster_model()
        summary["customer_segmentation"] = {
            "status": "Trained",
            "algorithm": "K-Means RFM + StandardScaler + PCA",
            "n_clusters": cust_cluster_bundle["n_clusters"],
            "silhouette_score": cust_cluster_bundle["final_silhouette"],
            "explained_variance": cust_cluster_bundle["explained_variance"],
            "k_values": cust_cluster_bundle["k_values"],
            "elbow_inertias": cust_cluster_bundle["elbow_inertias"],
            "silhouette_scores": cust_cluster_bundle["silhouette_scores"],
            "cluster_profiles": cust_cluster_bundle["cluster_labels_map"]
        }
    except Exception as e:
        summary["customer_segmentation"] = {"status": "Error", "message": str(e)}

    # 5. Anomaly Detection
    try:
        anomaly_bundle = load_anomaly_model()
        summary["anomaly_detection"] = {
            "status": "Trained",
            "algorithm": "Isolation Forest",
            "summary_stats": anomaly_bundle["summary_stats"]
        }
    except Exception as e:
        summary["anomaly_detection"] = {"status": "Error", "message": str(e)}

    return summary


if __name__ == "__main__":
    report = get_all_models_evaluation_summary()
    for model_name, info in report.items():
        print(f"[{model_name.upper()}]: {info.get('status')} - {info.get('model_type', info.get('algorithm', ''))}")
