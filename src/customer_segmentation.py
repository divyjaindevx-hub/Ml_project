"""
RetailIQ - Customer Segmentation Module
Implements RFM and behavioral customer segmentation using K-Means,
StandardScaler, PCA 2D mapping, and automated behavioral interpretation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.feature_engineering import build_customer_rfm_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CUSTOMER_CLUSTER_FILE = os.path.join(MODELS_DIR, "customer_cluster_model.pkl")

CUSTOMER_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Avg_Order_Value",
    "Avg_Discount"
]


def train_customer_segmentation(
    df: pd.DataFrame,
    k_range: tuple = (2, 8),
    n_clusters: int = 4,
    save_path: str = CUSTOMER_CLUSTER_FILE,
    random_state: int = 42
) -> dict:
    """
    Computes RFM features, standardizes metrics, computes Elbow and Silhouette evaluations,
    fits K-Means, projects clusters via PCA, and dynamically interprets segment personas.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[CUSTOMER SEGMENTATION] Building customer RFM features...")
    rfm_df = build_customer_rfm_features(df)

    X = rfm_df[CUSTOMER_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow and Silhouette evaluation
    elbow_inertias = []
    silhouette_scores = []
    k_values = list(range(k_range[0], min(k_range[1] + 1, len(rfm_df))))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        elbow_inertias.append(float(km.inertia_))
        score = float(silhouette_score(X_scaled, km.labels_))
        silhouette_scores.append(score)

    # Fit selected KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    final_silhouette = float(silhouette_score(X_scaled, cluster_labels))

    rfm_df["Cluster_ID"] = cluster_labels

    # PCA 2D coordinates for visual mapping
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    rfm_df["PCA1"] = np.round(pca_coords[:, 0], 3)
    rfm_df["PCA2"] = np.round(pca_coords[:, 1], 3)
    explained_variance = [float(v) for v in pca.explained_variance_ratio_]

    # Cluster-wise statistics
    cluster_stats = rfm_df.groupby("Cluster_ID")[CUSTOMER_FEATURES].mean().reset_index()

    # Dynamic Behavioral Interpretation
    overall_monetary = rfm_df["Monetary"].mean()
    overall_freq = rfm_df["Frequency"].mean()
    overall_recency = rfm_df["Recency"].mean()
    overall_discount = rfm_df["Avg_Discount"].mean()

    cluster_labels_map = {}
    for cid in range(n_clusters):
        row = cluster_stats[cluster_stats["Cluster_ID"] == cid].iloc[0]
        rec = row["Recency"]
        freq = row["Frequency"]
        mon = row["Monetary"]
        disc = row["Avg_Discount"]

        if mon > overall_monetary and rec < overall_recency:
            persona = "High-Value Champions"
            rec_strategy = "VIP rewards, early access to new releases, premium concierge service"
        elif disc > overall_discount * 1.15:
            persona = "Discount & Promotion Seekers"
            rec_strategy = "Targeted clearance sales, bundle savings, flash discounts"
        elif rec > overall_recency and freq < overall_freq:
            persona = "At-Risk / Lapsed Buyers"
            rec_strategy = "Win-back email campaigns, re-engagement incentives, feedback surveys"
        else:
            persona = "Steady Occasional Shoppers"
            rec_strategy = "Product recommendations, loyalty point incentives, free shipping thresholds"

        cluster_labels_map[cid] = {
            "name": f"Segment {cid}: {persona}",
            "persona": persona,
            "strategy": rec_strategy
        }

    rfm_df["Segment_Name"] = rfm_df["Cluster_ID"].map(lambda c: cluster_labels_map[c]["persona"])

    print(f"[CUSTOMER SEGMENTATION] Silhouette Score for k={n_clusters}: {final_silhouette:.4f}")
    for cid, meta in cluster_labels_map.items():
        count = int((rfm_df['Cluster_ID'] == cid).sum())
        print(f"  -> {meta['name']} ({count} customers)")

    model_bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "pca": pca,
        "features": CUSTOMER_FEATURES,
        "n_clusters": n_clusters,
        "k_values": k_values,
        "elbow_inertias": elbow_inertias,
        "silhouette_scores": silhouette_scores,
        "final_silhouette": round(final_silhouette, 4),
        "explained_variance": explained_variance,
        "cluster_stats": cluster_stats,
        "cluster_labels_map": cluster_labels_map,
        "clustered_customers": rfm_df
    }

    joblib.dump(model_bundle, save_path)
    print(f"[CUSTOMER SEGMENTATION] Saved customer cluster bundle to {save_path}")

    return model_bundle


def load_customer_cluster_model(model_path: str = CUSTOMER_CLUSTER_FILE) -> dict:
    """Loads saved customer cluster bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train customer segmentation first.")
    return joblib.load(model_path)


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import clean_retail_data

    raw = load_dataset()
    clean = clean_retail_data(raw)
    bundle = train_customer_segmentation(clean, n_clusters=4)
    print("Clustered customers sample:\n", bundle["clustered_customers"][["Customer ID", "Segment_Name", "Recency", "Frequency", "Monetary"]].head())
