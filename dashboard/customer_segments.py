"""
RetailIQ Dashboard - Page 5: Customer Segmentation & RFM Analytics
Displays customer clusters based on Recency, Frequency, and Monetary (RFM) behavior,
average order values, discount sensitivity, and prescriptive CRM retention strategies.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.customer_segmentation import load_customer_cluster_model


def render_customer_segments_page(df: pd.DataFrame, cust_cluster_bundle: dict = None):
    st.markdown("## 👥 Customer Segmentation & RFM Behavioral Intelligence")
    st.markdown(
        "Segment your retail customer base using Recency, Frequency, Monetary (RFM), and promotion sensitivity metrics. "
        "Align marketing spend and inventory replenishment to demand drivers."
    )

    if cust_cluster_bundle is None:
        try:
            cust_cluster_bundle = load_customer_cluster_model()
        except Exception as e:
            st.error(f"Customer cluster bundle not found: {e}. Please train customer segmentation first.")
            return

    rfm_df = cust_cluster_bundle["clustered_customers"]
    cluster_labels_map = cust_cluster_bundle["cluster_labels_map"]
    silhouette = cust_cluster_bundle["final_silhouette"]
    explained_var = cust_cluster_bundle["explained_variance"]

    # 1. High-Level Metrics
    total_custs = len(rfm_df)
    avg_spend = rfm_df["Monetary"].mean()
    avg_freq = rfm_df["Frequency"].mean()
    avg_rec = rfm_df["Recency"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers Evaluated", f"{total_custs:,}")
    with c2:
        st.metric("Average Customer Lifetime Spend", f"${avg_spend:,.2f}")
    with c3:
        st.metric("Average Purchase Frequency", f"{avg_freq:.1f} orders")
    with c4:
        st.metric("Average Recency", f"{avg_rec:.1f} days ago")

    st.markdown("---")

    # 2. Charts Row 1: Frequency vs. Monetary Spend & 2D PCA Map
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        st.subheader("💳 Spend vs. Purchase Frequency")
        st.caption("Customer distribution highlighting high-value VIPs vs. infrequent shoppers")

        fig_scatter = px.scatter(
            rfm_df,
            x="Frequency",
            y="Monetary",
            color="Segment_Name",
            size="Avg_Order_Value",
            hover_name="Customer ID",
            hover_data=["Recency", "Avg_Order_Value", "Avg_Discount", "Preferred_Category"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"Frequency": "Total Transactions (Frequency)", "Monetary": "Total Expenditure ($)"}
        )
        fig_scatter.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with r1_col2:
        st.subheader("🗺️ 2D PCA RFM Cluster Projection")
        st.caption(f"Dimensionality reduction explaining {(explained_var[0]+explained_var[1])*100:.1f}% of customer RFM variance")

        fig_pca = px.scatter(
            rfm_df,
            x="PCA1",
            y="PCA2",
            color="Segment_Name",
            hover_name="Customer ID",
            hover_data=["Monetary", "Frequency", "Recency"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"}
        )
        fig_pca.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")

    # 3. Charts Row 2: Customer Segment Distribution & Discount Sensitivity
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.subheader("📊 Customer Segment Proportions")
        seg_counts = rfm_df["Segment_Name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig_pie = px.pie(
            seg_counts, names="Segment", values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with r2_col2:
        st.subheader("🏷️ Average Order Value (AOV) by Segment")
        aov_df = rfm_df.groupby("Segment_Name")["Avg_Order_Value"].mean().reset_index()

        fig_aov = px.bar(
            aov_df, x="Segment_Name", y="Avg_Order_Value",
            color="Segment_Name",
            text="Avg_Order_Value",
            labels={"Avg_Order_Value": "Average Order Value ($)", "Segment_Name": ""},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_aov.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
        fig_aov.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320
        )
        st.plotly_chart(fig_aov, use_container_width=True)

    st.markdown("---")

    # 4. Cluster Profile & Prescriptive CRM Strategies
    st.subheader("📋 Segment Personas & Business Action Playbook")

    stats = cust_cluster_bundle["cluster_stats"].copy()
    stats["Segment_Persona"] = stats["Cluster_ID"].map(lambda c: cluster_labels_map[c]["persona"])
    stats["Prescriptive_Retention_Strategy"] = stats["Cluster_ID"].map(lambda c: cluster_labels_map[c]["strategy"])
    stats["Customer_Count"] = stats["Cluster_ID"].map(lambda c: int((rfm_df["Cluster_ID"] == c).sum()))

    st.dataframe(
        stats[[
            "Cluster_ID", "Segment_Persona", "Customer_Count",
            "Recency", "Frequency", "Monetary", "Avg_Order_Value",
            "Avg_Discount", "Prescriptive_Retention_Strategy"
        ]].style.format({
            "Recency": "{:.1f} days",
            "Frequency": "{:.1f} orders",
            "Monetary": "${:,.2f}",
            "Avg_Order_Value": "${:,.2f}",
            "Avg_Discount": "{:.1%}",
            "Customer_Count": "{:,}"
        }),
        use_container_width=True,
        hide_index=True
    )
