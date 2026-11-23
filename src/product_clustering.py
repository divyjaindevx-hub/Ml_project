"""
RetailIQ - Product Clustering Module
Implements K-Means clustering for multidimensional product segmentation,
evaluating inertia via Elbow Method, Silhouette scores, PCA 2D projections,
and automated data-driven cluster profiling without hardcoded labels.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.feature_engineering import build_product_clustering_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PRODUCT_CLUSTER_FILE = os.path.join(MODELS_DIR, "product_cluster_model.pkl")

PRODUCT_CLUSTER_FEATURES = [
    "Avg_Daily_Sales",
    "Total_Revenue",
    "Profit_Margin_Pct",
    "Stock_Turnover",
    "Demand_Variability",
    "Current_Stock"
]


def train_product_clustering(
    df: pd.DataFrame,
    k_range: tuple = (2, 8),
    n_clusters: int = 4,
    save_path: str = PRODUCT_CLUSTER_FILE,
    random_state: int = 42
) -> dict:
    """
    Computes product summary metrics, scales features, runs Elbow and Silhouette analysis,
    fits optimal K-Means clustering, computes PCA 2D projections, and dynamically
    interprets cluster characteristics based on quantitative rankings.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[PRODUCT CLUSTERING] Building product-level performance features...")
    prod_df = build_product_clustering_features(df)

    X = prod_df[PRODUCT_CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute Elbow curve (inertia) and Silhouette scores across k_range
    elbow_inertias = []
    silhouette_scores = []
    k_values = list(range(k_range[0], min(k_range[1] + 1, len(prod_df))))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        elbow_inertias.append(float(km.inertia_))
        score = float(silhouette_score(X_scaled, km.labels_))
        silhouette_scores.append(score)

    # Fit chosen KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    final_silhouette = float(silhouette_score(X_scaled, cluster_labels))

    prod_df["Cluster_ID"] = cluster_labels

    # PCA 2D projection
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    prod_df["PCA1"] = np.round(pca_coords[:, 0], 3)
    prod_df["PCA2"] = np.round(pca_coords[:, 1], 3)
    explained_variance = [float(v) for v in pca.explained_variance_ratio_]

    # Cluster-wise statistical profiling (Mean values per cluster)
    cluster_stats = prod_df.groupby("Cluster_ID")[PRODUCT_CLUSTER_FEATURES].mean().reset_index()

    # Dynamic Data-Driven Cluster Labeling based on actual characteristics
    cluster_labels_map = {}
    for cid in range(n_clusters):
        row = cluster_stats[cluster_stats["Cluster_ID"] == cid].iloc[0]
        sales = row["Avg_Daily_Sales"]
        rev = row["Total_Revenue"]
        margin = row["Profit_Margin_Pct"]
        turnover = row["Stock_Turnover"]
        stock = row["Current_Stock"]

        overall_sales_mean = prod_df["Avg_Daily_Sales"].mean()
        overall_turnover_mean = prod_df["Stock_Turnover"].mean()
        overall_margin_mean = prod_df["Profit_Margin_Pct"].mean()
        overall_stock_mean = prod_df["Current_Stock"].mean()

        tags = []
        if sales > overall_sales_mean and turnover > overall_turnover_mean:
            name = "Fast-Moving & High-Velocity"
            tags.append("High turnover, critical stock replenishment needed")
        elif rev > prod_df["Total_Revenue"].median() and margin > overall_margin_mean:
            name = "High-Margin Cash Cows"
            tags.append("Strong profitability, prioritize customer availability")
        elif sales < overall_sales_mean and stock > overall_stock_mean:
            name = "Slow-Moving / Overstock Risk"
            tags.append("Low turnover, consider promotional clearance")
        else:
            name = "Core Steady Performers"
            tags.append("Balanced velocity and stable inventory demand")

        cluster_labels_map[cid] = {
            "name": f"Cluster {cid}: {name}",
            "short_name": name,
            "description": tags[0] if tags else "Standard performance product group"
        }

    prod_df["Cluster_Name"] = prod_df["Cluster_ID"].map(lambda c: cluster_labels_map[c]["short_name"])

    print(f"[PRODUCT CLUSTERING] Optimal Silhouette Score for k={n_clusters}: {final_silhouette:.4f}")
    for cid, meta in cluster_labels_map.items():
        count = int((prod_df['Cluster_ID'] == cid).sum())
        print(f"  -> {meta['name']} ({count} products)")

    model_bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "pca": pca,
        "features": PRODUCT_CLUSTER_FEATURES,
        "n_clusters": n_clusters,
        "k_values": k_values,
        "elbow_inertias": elbow_inertias,
        "silhouette_scores": silhouette_scores,
        "final_silhouette": round(final_silhouette, 4),
        "explained_variance": explained_variance,
        "cluster_stats": cluster_stats,
        "cluster_labels_map": cluster_labels_map,
        "clustered_products": prod_df
    }

    joblib.dump(model_bundle, save_path)
    print(f"[PRODUCT CLUSTERING] Saved product cluster bundle to {save_path}")

    return model_bundle


def load_product_cluster_model(model_path: str = PRODUCT_CLUSTER_FILE) -> dict:
    """Loads saved product clustering model bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train product clustering first.")
    return joblib.load(model_path)


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import clean_retail_data

    raw = load_dataset()
    clean = clean_retail_data(raw)
    bundle = train_product_clustering(clean, n_clusters=4)
    print("Clustered products sample:\n", bundle["clustered_products"][["Product ID", "Product_Name", "Cluster_Name", "Avg_Daily_Sales", "Stock_Turnover"]].head())
