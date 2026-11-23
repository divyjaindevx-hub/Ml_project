"""
RetailIQ Dashboard - Page 3: Inventory Intelligence & Prescriptive Engine
Displays current inventory health, automated reorder recommendations,
stockout probabilities, supplier lead time diagnostics, and inventory turnover.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.inventory_engine import evaluate_product_inventory_status
from src.stockout_prediction import load_stockout_model


def render_inventory_page(df: pd.DataFrame, stockout_bundle: dict = None):
    st.markdown("## 📦 Inventory Intelligence & Prescriptive Engine")
    st.markdown(
        "Synthesize supply chain science and predictive machine learning to detect stockout vulnerabilities, "
        "identify stagnant capital, and optimize replenishment schedules."
    )

    if stockout_bundle is None:
        try:
            stockout_bundle = load_stockout_model()
        except Exception:
            stockout_bundle = None

    inventory_plan = evaluate_product_inventory_status(df, stockout_bundle)

    # 1. High-Level Summary Metrics
    total_skus = len(inventory_plan)
    urgent_skus = int((inventory_plan["Action_Type"] == "Urgent Action Required").sum())
    excess_skus = int((inventory_plan["Action_Type"] == "Excess Capital").sum())
    total_val = inventory_plan["Inventory_Value"].sum()
    total_reorder_qty = inventory_plan["Suggested_Reorder_Qty"].sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Catalog SKUs", f"{total_skus:,}")
    with m2:
        st.metric("Total Warehouse Value", f"${total_val:,.0f}")
    with m3:
        st.metric("Immediate Reorders", f"{urgent_skus} SKUs", delta="Critical attention", delta_color="inverse")
    with m4:
        st.metric("Overstock SKUs", f"{excess_skus} SKUs", delta="Stagnant capital", delta_color="off")
    with m5:
        st.metric("Total Reorder Units", f"{total_reorder_qty:,} units")

    st.markdown("---")

    # 2. Charts: Inventory Turnover vs Margin & Supplier Lead Time Analysis
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("🎯 Stock Turnover vs. Profit Margin Matrix")
        st.caption("Quadrant mapping to identify Fast-Moving Cash Cows vs. Slow-Moving Stagnant Stock")

        fig_scatter = px.scatter(
            inventory_plan,
            x="Stock_Turnover",
            y="Profit_Margin_Pct",
            size="Inventory_Value",
            color="ML_Stockout_Risk",
            hover_name="Product_Name",
            hover_data=["Category", "Current_Stock", "Stock_Cover_Days", "Primary_Recommendation"],
            color_discrete_map={"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"},
            labels={"Stock_Turnover": "Stock Turnover Ratio", "Profit_Margin_Pct": "Profit Margin (%)"}
        )
        fig_scatter.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_c2:
        st.subheader("🚚 Supplier Lead Time & Risk Exposure")
        st.caption("Average lead time and high-risk inventory exposure across suppliers")

        sup_agg = inventory_plan.groupby("Supplier").agg(
            Avg_Lead_Time=("Lead_Time", "mean"),
            High_Risk_SKUs=("ML_Stockout_Risk", lambda r: (r == "High Risk").sum()),
            Total_Value=("Inventory_Value", "sum")
        ).reset_index()

        fig_sup = px.bar(
            sup_agg,
            x="Supplier",
            y="Avg_Lead_Time",
            color="High_Risk_SKUs",
            color_continuous_scale="Reds",
            text="Avg_Lead_Time",
            labels={"Avg_Lead_Time": "Avg Lead Time (Days)", "High_Risk_SKUs": "High Risk Count"}
        )
        fig_sup.update_traces(texttemplate="%{text:.1f} d", textposition="outside")
        fig_sup.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_sup, use_container_width=True)

    st.markdown("---")

    # 3. Interactive Inventory Table with Filters
    st.subheader("📋 Interactive SKU Inventory Health Table")

    f1, f2, f3 = st.columns(3)
    with f1:
        cat_filter = st.multiselect("Filter by Category", inventory_plan["Category"].unique().tolist(), default=inventory_plan["Category"].unique().tolist())
    with f2:
        risk_filter = st.multiselect("Filter by Stockout Risk", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk", "Low Risk"])
    with f3:
        search_query = st.text_input("Search Product ID or Name", "")

    filtered = inventory_plan[
        (inventory_plan["Category"].isin(cat_filter)) &
        (inventory_plan["ML_Stockout_Risk"].isin(risk_filter))
    ]

    if search_query:
        filtered = filtered[
            filtered["Product ID"].str.contains(search_query, case=False) |
            filtered["Product_Name"].str.contains(search_query, case=False)
        ]

    display_cols = [
        "Product ID", "Product_Name", "Category", "Current_Stock",
        "Calculated_Reorder_Point", "Stock_Cover_Days", "ML_Stockout_Risk",
        "Stockout_Probability", "Stock_Turnover", "Inventory_Value",
        "Primary_Recommendation", "Suggested_Reorder_Qty"
    ]

    st.dataframe(
        filtered[display_cols].style.format({
            "Inventory_Value": "${:,.2f}",
            "Stockout_Probability": "{:.1%}",
            "Stock_Cover_Days": "{:.1f} days",
            "Stock_Turnover": "{:.1f}x",
            "Suggested_Reorder_Qty": "{:,}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Download CSV button
    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Inventory Plan to CSV",
        data=csv_bytes,
        file_name="RetailIQ_Inventory_Prescriptions.csv",
        mime="text/csv"
    )
