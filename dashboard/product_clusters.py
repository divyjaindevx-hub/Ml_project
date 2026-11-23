"""
RetailIQ Dashboard - Page 4: Product Clustering & Performance Taxonomy
Presents multi-attribute K-Means product segmentation, PCA 2D projections,
Elbow method inertia curves, Silhouette diagnostic scores, and cluster characteristics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.product_clustering import load_product_cluster_model


def render_product_clusters_page(df: pd.DataFrame, prod_cluster_bundle: dict = None):
    st.markdown("## 🧩 Product Clustering & Inventory Taxonomy")
    st.markdown(
        "Discover hidden behavioral groupings across your product catalog using unsupervised K-Means clustering. "
        "Segments are categorized by actual turnover, revenue contribution, profit margins, and inventory volatility."
    )

    if prod_cluster_bundle is None:
        try:
            prod_cluster_bundle = load_product_cluster_model()
        except Exception as e:
            st.error(f"Product clustering model not loaded: {e}. Train the product clustering model first.")
            return

    clustered_df = prod_cluster_bundle["clustered_products"]
    cluster_labels_map = prod_cluster_bundle["cluster_labels_map"]
    silhouette = prod_cluster_bundle["final_silhouette"]
    explained_var = prod_cluster_bundle["explained_variance"]

    # 1. Overview KPIs
    k_val = prod_cluster_bundle["n_clusters"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Clusters Identified", f"{k_val}")
    with c2:
        st.metric("Silhouette Score", f"{silhouette:.4f}", delta="Optimal partition", delta_color="normal")
    with c3:
        st.metric("PCA 2D Variance", f"{(explained_var[0]+explained_var[1])*100:.1f}%")
    with c4:
        st.metric("Total Products Evaluated", f"{len(clustered_df)}")

    st.markdown("---")

    # 2. Charts Row 1: PCA 2D Projection & Cluster Distribution
    r1_col1, r1_col2 = st.columns([7, 5])

    with r1_col1:
        st.subheader("🗺️ 2D PCA Dimensionality Projection")
        st.caption("Principal components capturing primary product variance in sales, turnover, and margins")

        fig_pca = px.scatter(
            clustered_df,
            x="PCA1",
            y="PCA2",
            color="Cluster_Name",
            hover_name="Product_Name",
            hover_data=["Category", "Avg_Daily_Sales", "Stock_Turnover", "Profit_Margin_Pct", "Total_Revenue"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"PCA1": f"Principal Component 1 ({explained_var[0]*100:.1f}%)", "PCA2": f"Principal Component 2 ({explained_var[1]*100:.1f}%)"}
        )
        fig_pca.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")))
        fig_pca.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with r1_col2:
        st.subheader("📊 Product Count by Cluster")
        cluster_counts = clustered_df["Cluster_Name"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]

        fig_bar = px.bar(
            cluster_counts,
            x="Count",
            y="Cluster",
            orientation="h",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bar.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # 3. Model Diagnostics: Elbow Curve & Silhouette Analysis
    st.subheader("📐 Clustering Diagnostics & Hyperparameter Validation")
    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("##### 📉 Elbow Method (Inertia vs. k)")
        k_values = prod_cluster_bundle["k_values"]
        inertias = prod_cluster_bundle["elbow_inertias"]

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_values, y=inertias,
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=8, color="#1D4ED8")
        ))
        # Highlight selected k
        if k_val in k_values:
            idx = k_values.index(k_val)
            fig_elbow.add_trace(go.Scatter(
                x=[k_val], y=[inertias[idx]],
                mode="markers",
                marker=dict(size=14, color="#EF4444", symbol="star"),
                name=f"Selected k={k_val}"
            ))
        fig_elbow.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (Sum of Squared Distances)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with diag_c2:
        st.markdown("##### 🌟 Silhouette Score vs. k")
        sil_scores = prod_cluster_bundle["silhouette_scores"]

        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=k_values, y=sil_scores,
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=8, color="#059669")
        ))
        if k_val in k_values:
            idx = k_values.index(k_val)
            fig_sil.add_trace(go.Scatter(
                x=[k_val], y=[sil_scores[idx]],
                mode="markers",
                marker=dict(size=14, color="#EF4444", symbol="star"),
                name=f"Selected k={k_val}"
            ))
        fig_sil.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Coefficient",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # 4. Cluster Characteristics Profiling Table
    st.subheader("📋 Quantitative Cluster Behavioral Profiles")
    st.markdown(
        "Each cluster represents a distinctive operational archetype based on its actual quantitative averages."
    )

    cluster_stats = prod_cluster_bundle["cluster_stats"].copy()
    cluster_stats["Cluster_Name"] = cluster_stats["Cluster_ID"].map(
        lambda c: cluster_labels_map[c]["short_name"]
    )
    cluster_stats["Recommended_Strategy"] = cluster_stats["Cluster_ID"].map(
        lambda c: cluster_labels_map[c]["description"]
    )

    st.dataframe(
        cluster_stats[[
            "Cluster_ID", "Cluster_Name", "Avg_Daily_Sales", "Total_Revenue",
            "Profit_Margin_Pct", "Stock_Turnover", "Demand_Variability", "Recommended_Strategy"
        ]].style.format({
            "Avg_Daily_Sales": "{:.1f} units/day",
            "Total_Revenue": "${:,.0f}",
            "Profit_Margin_Pct": "{:.1f}%",
            "Stock_Turnover": "{:.1f}x",
            "Demand_Variability": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # 5. Product-Level Cluster Assignments
    st.subheader("🔍 Individual SKU Cluster Assignments")
    sel_cluster = st.selectbox("Filter by Cluster Persona:", ["All"] + list(clustered_df["Cluster_Name"].unique()))

    prod_view = clustered_df.copy()
    if sel_cluster != "All":
        prod_view = prod_view[prod_view["Cluster_Name"] == sel_cluster]

    st.dataframe(
        prod_view[[
            "Product ID", "Product_Name", "Category", "Cluster_Name",
            "Avg_Daily_Sales", "Total_Revenue", "Stock_Turnover",
            "Profit_Margin_Pct", "Current_Stock"
        ]].style.format({
            "Avg_Daily_Sales": "{:.1f}",
            "Total_Revenue": "${:,.2f}",
            "Stock_Turnover": "{:.1f}x",
            "Profit_Margin_Pct": "{:.1f}%",
            "Current_Stock": "{:,}"
        }),
        use_container_width=True,
        hide_index=True
    )
