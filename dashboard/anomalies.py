"""
RetailIQ Dashboard - Page 6: Anomaly Detection & Operational Risk
Presents Isolation Forest anomaly detection results, normal vs. anomalous
sales timelines, category risk distribution, and natural language explanations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.anomaly_detection import load_anomaly_model


def render_anomalies_page(df: pd.DataFrame, anomaly_bundle: dict = None):
    st.markdown("## 🚨 Anomaly Detection & Operational Integrity")
    st.markdown(
        "Detect subtle disruptions, unexpected sales spikes, sudden ghost demand drop-offs, and critical stockout "
        "events using unsupervised Isolation Forest machine learning."
    )

    if anomaly_bundle is None:
        try:
            anomaly_bundle = load_anomaly_model()
        except Exception as e:
            st.error(f"Anomaly model not loaded: {e}. Train the anomaly detection model first.")
            return

    summary_stats = anomaly_bundle["summary_stats"]
    anomalies_df = anomaly_bundle["anomalies_df"]
    enriched_df = anomaly_bundle["enriched_df"]

    # 1. Anomaly KPI Cards
    total_records = summary_stats["total_records"]
    anomaly_count = summary_stats["anomaly_count"]
    normal_count = summary_stats["normal_count"]
    anomaly_pct = summary_stats["anomaly_pct"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records Inspected", f"{total_records:,}")
    with c2:
        st.metric("Normal Transactions", f"{normal_count:,}")
    with c3:
        st.metric("Detected Anomalies", f"{anomaly_count:,}", delta=f"{anomaly_pct}% anomaly rate", delta_color="inverse")
    with c4:
        st.metric("Model Contamination Setting", f"{summary_stats['contamination']*100:.1f}%")

    st.markdown("---")

    # 2. Charts Row 1: Interactive Timeline of Normal vs. Anomalous Sales
    st.subheader("📈 Sales Activity Timeline: Inliers vs. Detected Anomalies")

    # Filter by product or view aggregate
    products = enriched_df[["Product ID", "Product Name"]].drop_duplicates().sort_values("Product ID")
    prod_options = ["All Products (Aggregate)"] + [f"{r['Product ID']} – {r['Product Name']}" for _, r in products.iterrows()]
    sel_prod_option = st.selectbox("Filter Timeline by Product:", prod_options, index=0)

    timeline_data = enriched_df.copy()
    timeline_data["Sales Date"] = pd.to_datetime(timeline_data["Sales Date"])

    if sel_prod_option != "All Products (Aggregate)":
        target_pid = sel_prod_option.split(" – ")[0]
        timeline_data = timeline_data[timeline_data["Product ID"] == target_pid]

    normal_points = timeline_data[timeline_data["Is_Anomaly"] == 0]
    anomaly_points = timeline_data[timeline_data["Is_Anomaly"] == 1]

    fig_timeline = go.Figure()

    # Normal records
    fig_timeline.add_trace(go.Scatter(
        x=normal_points["Sales Date"], y=normal_points["Units Sold"],
        mode="markers", name="Normal Transaction",
        marker=dict(color="#3B82F6", size=4, opacity=0.45)
    ))

    # Anomalies
    fig_timeline.add_trace(go.Scatter(
        x=anomaly_points["Sales Date"], y=anomaly_points["Units Sold"],
        mode="markers", name="Detected Anomaly",
        marker=dict(color="#EF4444", size=9, symbol="circle-open-dot", line=dict(width=2, color="#B91C1C")),
        text=anomaly_points["Anomaly_Reason"],
        hovertemplate="<b>Date:</b> %{x}<br><b>Units:</b> %{y}<br><b>Reason:</b> %{text}<extra></extra>"
    ))

    fig_timeline.update_layout(
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Sales Date",
        yaxis_title="Units Sold",
        legend=dict(orientation="h", y=1.12),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")

    # 3. Charts Row 2: Anomaly Breakdown by Category & Score Distribution
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        st.subheader("🛍️ Anomaly Occurrence by Category")
        cat_anom = anomalies_df["Category"].value_counts().reset_index()
        cat_anom.columns = ["Category", "Anomaly_Count"]

        fig_cat = px.bar(
            cat_anom, x="Category", y="Anomaly_Count",
            color="Category",
            text="Anomaly_Count",
            labels={"Anomaly_Count": "Anomalies Count", "Category": ""},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_cat.update_traces(textposition="outside")
        fig_cat.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with r2_c2:
        st.subheader("📊 Anomaly Severity Score Distribution")
        fig_hist = px.histogram(
            anomalies_df, x="Anomaly_Score",
            nbins=30,
            color_discrete_sequence=["#EF4444"],
            labels={"Anomaly_Score": "Anomaly Severity Score (0 to 1)"}
        )
        fig_hist.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Severity Score (Higher = More Extreme)",
            yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # 4. Detailed Anomaly Inspection Table
    st.subheader("🔍 Detected Anomalies Register & Root-Cause Explanations")

    min_score = st.slider("Minimum Anomaly Severity Threshold:", min_value=0.50, max_value=1.00, value=0.65, step=0.05)
    filtered_anomalies = anomalies_df[anomalies_df["Anomaly_Score"] >= min_score].copy()

    display_cols = [
        "Sales Date", "Product ID", "Product Name", "Category",
        "Units Sold", "Current Stock", "Discount",
        "Anomaly_Score", "Anomaly_Reason"
    ]

    st.dataframe(
        filtered_anomalies[display_cols].style.format({
            "Anomaly_Score": "{:.3f}",
            "Discount": "{:.0%}",
            "Units Sold": "{:,}",
            "Current Stock": "{:,}"
        }),
        use_container_width=True,
        hide_index=True
    )
