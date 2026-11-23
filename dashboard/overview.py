"""
RetailIQ Dashboard - Page 1: Executive Overview
Displays high-level inventory KPIs, revenue trends, category distributions,
top-performing products, monthly financial summaries, and stock health metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_overview_page(df: pd.DataFrame, inventory_df: pd.DataFrame, eda_summary: dict):
    st.markdown("## 📊 Executive Overview & Retail KPIs")
    st.markdown(
        "A holistic executive summary of enterprise retail performance, current warehouse valuation, "
        "sales momentum, and critical inventory health indicators."
    )

    # 1. Top KPI Metric Cards
    total_products = eda_summary.get("total_products", 0)
    inventory_value = eda_summary.get("current_inventory_value", 0.0)
    total_rev = eda_summary.get("total_revenue", 0.0)
    total_units = eda_summary.get("total_units_sold", 0)
    avg_margin = eda_summary.get("avg_profit_margin_pct", 0.0)

    high_risk_count = 0
    if inventory_df is not None and "ML_Stockout_Risk" in inventory_df.columns:
        high_risk_count = int((inventory_df["ML_Stockout_Risk"] == "High Risk").sum())
    elif inventory_df is not None and "Stockout_Probability" in inventory_df.columns:
        high_risk_count = int((inventory_df["Stockout_Probability"] >= 0.7).sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(label="Total SKUs", value=f"{total_products:,}")
    with c2:
        st.metric(label="Inventory Value", value=f"${inventory_value:,.0f}")
    with c3:
        st.metric(label="Total Revenue", value=f"${total_rev:,.0f}")
    with c4:
        st.metric(label="Units Sold", value=f"{total_units:,}")
    with c5:
        st.metric(label="Avg Profit Margin", value=f"{avg_margin:.1f}%")
    with c6:
        st.metric(label="High Risk SKUs", value=f"{high_risk_count}", delta=f"{high_risk_count} urgent", delta_color="inverse")

    st.markdown("---")

    # 2. Charts Row 1: Revenue Trends & Category Share
    r1_col1, r1_col2 = st.columns([7, 5])

    with r1_col1:
        st.subheader("📈 Revenue & Sales Momentum Over Time")
        time_freq = st.radio("Aggregation Interval", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="ov_freq")

        df_time = df.copy()
        df_time["Sales Date"] = pd.to_datetime(df_time["Sales Date"])
        freq_code = "D" if time_freq == "Daily" else ("W" if time_freq == "Weekly" else "ME")

        trend = df_time.groupby(pd.Grouper(key="Sales Date", freq=freq_code)).agg(
            Revenue=("Revenue", "sum") if "Revenue" in df_time.columns else ("Units Sold", lambda s: (s * df_time.loc[s.index, "Unit Price"]).sum()),
            Units=("Units Sold", "sum")
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend["Sales Date"], y=trend["Revenue"],
            name="Revenue ($)", mode="lines",
            line=dict(color="#0284C7", width=2.5),
            fill="tozeroy", fillcolor="rgba(2, 132, 199, 0.1)"
        ))
        fig_trend.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with r1_col2:
        st.subheader("🛍️ Sales Revenue by Category")
        cat_rev = df.groupby("Category")["Revenue" if "Revenue" in df.columns else "Units Sold"].sum().reset_index()
        cat_rev.columns = ["Category", "Amount"]

        fig_pie = px.pie(
            cat_rev, names="Category", values="Amount",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", y=-0.15),
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # 3. Charts Row 2: Top Selling Products & Inventory Distribution
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.subheader("🏆 Top-Selling Products by Revenue")
        top_prods = df.groupby(["Product ID", "Product Name"])["Revenue" if "Revenue" in df.columns else "Units Sold"].sum().reset_index()
        top_prods.columns = ["Product ID", "Product Name", "Sales_Metric"]
        top_prods = top_prods.sort_values("Sales_Metric", ascending=True).tail(8)

        fig_top = px.bar(
            top_prods, x="Sales_Metric", y="Product Name",
            orientation="h",
            labels={"Sales_Metric": "Total Revenue ($)", "Product Name": ""},
            color="Sales_Metric",
            color_continuous_scale="Blues"
        )
        fig_top.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with r2_col2:
        st.subheader("📦 Warehouse Inventory Value by Category")
        if inventory_df is not None and "Inventory_Value" in inventory_df.columns:
            cat_inv = inventory_df.groupby("Category").agg(
                Total_Value=("Inventory_Value", "sum"),
                Units=("Current_Stock", "sum")
            ).reset_index()

            fig_inv = px.bar(
                cat_inv, x="Category", y="Total_Value",
                text="Total_Value",
                labels={"Total_Value": "Inventory Valuation ($)", "Category": ""},
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_inv.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_inv.update_layout(
                margin=dict(l=10, r=10, t=25, b=10),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=340
            )
            st.plotly_chart(fig_inv, use_container_width=True)
        else:
            st.info("Inventory valuation breakdown will appear once inventory calculations are loaded.")

    st.markdown("---")

    # 4. Charts Row 3: Monthly Performance & Stock Health Risk Overview
    r3_col1, r3_col2 = st.columns([7, 5])

    with r3_col1:
        st.subheader("📅 Monthly Performance & Gross Profit Breakdown")
        df_month = df.copy()
        df_month["Sales Date"] = pd.to_datetime(df_month["Sales Date"])
        df_month["Year-Month"] = df_month["Sales Date"].dt.to_period("M").astype(str)

        monthly = df_month.groupby("Year-Month").agg(
            Revenue=("Revenue", "sum") if "Revenue" in df_month.columns else ("Units Sold", lambda s: (s * df_month.loc[s.index, "Unit Price"]).sum()),
            Profit=("Profit", "sum") if "Profit" in df_month.columns else ("Revenue", lambda s: s.sum() * 0.4)
        ).reset_index()

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly["Year-Month"], y=monthly["Revenue"],
            name="Revenue ($)", marker_color="#3B82F6"
        ))
        fig_monthly.add_trace(go.Bar(
            x=monthly["Year-Month"], y=monthly["Profit"],
            name="Gross Profit ($)", marker_color="#10B981"
        ))
        fig_monthly.update_layout(
            barmode="group",
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", y=1.12),
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with r3_col2:
        st.subheader("🛡️ Stock Health & Stockout Risk Overview")
        if inventory_df is not None and "ML_Stockout_Risk" in inventory_df.columns:
            risk_counts = inventory_df["ML_Stockout_Risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Level", "Count"]

            color_map = {"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"}

            fig_risk = px.pie(
                risk_counts, names="Risk_Level", values="Count",
                color="Risk_Level",
                color_discrete_map=color_map,
                hole=0.4
            )
            fig_risk.update_layout(
                margin=dict(l=10, r=10, t=25, b=10),
                legend=dict(orientation="h", y=-0.15),
                height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Stock health distribution will appear once stockout classification model is trained.")
