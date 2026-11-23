"""
RetailIQ – Smart Inventory Intelligence & Predictive Analytics Platform
Main Streamlit Application Entrypoint
"""

import os
import streamlit as st
import pandas as pd

# Set page configuration must be first Streamlit call
st.set_page_config(
    page_title="RetailIQ – Smart Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Style CSS
st.markdown("""
<style>
    /* Global style adjustments */
    .main {
        background-color: #0F172A;
    }
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    /* Metric Card styling */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #0284C7;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Sidebar branding */
    .sidebar-brand {
        padding: 10px 0px 20px 0px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 15px;
    }
    .badge-synthetic {
        background-color: rgba(59, 130, 246, 0.15);
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
        display: inline-block;
        margin-top: 6px;
    }
    .badge-custom {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(52, 211, 153, 0.3);
        display: inline-block;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src and dashboard
from src.data_loader import load_dataset
from src.preprocessing import clean_retail_data, calculate_eda_summary
from src.demand_forecasting import load_demand_model, train_demand_model
from src.stockout_prediction import load_stockout_model, train_stockout_model
from src.product_clustering import load_product_cluster_model, train_product_clustering
from src.customer_segmentation import load_customer_cluster_model, train_customer_segmentation
from src.anomaly_detection import load_anomaly_model, train_anomaly_model
from src.inventory_engine import evaluate_product_inventory_status
from src.model_evaluation import get_all_models_evaluation_summary

from dashboard.overview import render_overview_page
from dashboard.forecasting import render_forecasting_page
from dashboard.inventory import render_inventory_page
from dashboard.product_clusters import render_product_clusters_page
from dashboard.customer_segments import render_customer_segments_page
from dashboard.anomalies import render_anomalies_page
from dashboard.model_performance import render_model_performance_page
from dashboard.data_explorer import render_data_explorer_page


@st.cache_data
def get_clean_default_data():
    """Loads and caches the default retail dataset."""
    raw = load_dataset()
    cleaned = clean_retail_data(raw)
    return cleaned


@st.cache_resource
def get_all_models():
    """Loads and caches all machine learning model bundles."""
    models = {}
    try:
        models["demand"] = load_demand_model()
    except Exception:
        models["demand"] = None

    try:
        models["stockout"] = load_stockout_model()
    except Exception:
        models["stockout"] = None

    try:
        models["prod_cluster"] = load_product_cluster_model()
    except Exception:
        models["prod_cluster"] = None

    try:
        models["cust_cluster"] = load_customer_cluster_model()
    except Exception:
        models["cust_cluster"] = None

    try:
        models["anomaly"] = load_anomaly_model()
    except Exception:
        models["anomaly"] = None

    return models


def retrain_all_models(df: pd.DataFrame):
    """Retrains all ML pipelines on the provided dataset."""
    with st.spinner("Retraining Demand Regressor..."):
        train_demand_model(df)
    with st.spinner("Retraining Stockout Risk Classifier..."):
        train_stockout_model(df)
    with st.spinner("Re-evaluating Product Clustering..."):
        train_product_clustering(df)
    with st.spinner("Re-evaluating Customer Segmentation..."):
        train_customer_segmentation(df)
    with st.spinner("Refitting Isolation Forest Anomaly Detection..."):
        train_anomaly_model(df)
    st.cache_resource.clear()
    st.success("All models successfully retrained and updated!")


def main():
    # Session state initialization
    if "active_df" not in st.session_state:
        st.session_state["active_df"] = get_clean_default_data()
        st.session_state["dataset_source"] = "Default Synthetic Retail Dataset"

    active_df = st.session_state["active_df"]
    models = get_all_models()

    # Precalculate EDA & Inventory metrics
    eda_summary = calculate_eda_summary(active_df)
    inventory_plan = evaluate_product_inventory_status(active_df, models.get("stockout"))

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2 style='margin:0; color:#38BDF8;'>RetailIQ</h2>
                <p style='margin:0; font-size:0.8rem; color:#94A3B8;'>Smart Inventory Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Dataset Status Indicator
        is_synthetic = "Synthetic" in st.session_state["dataset_source"]
        if is_synthetic:
            st.markdown(
                f"<span class='badge-synthetic'>⚙️ {st.session_state['dataset_source']}</span>",
                unsafe_allow_html=True
            )
            st.caption("Realistic simulation (730 days, 50 SKUs, 350 customers)")
        else:
            st.markdown(
                f"<span class='badge-custom'>📁 {st.session_state['dataset_source']}</span>",
                unsafe_allow_html=True
            )

        st.markdown("### Navigation")
        pages = [
            "Executive Overview",
            "Demand Forecasting",
            "Inventory Intelligence",
            "Product Clustering",
            "Customer Segmentation",
            "Anomaly Detection",
            "Model Performance",
            "Data Explorer"
        ]
        page_icons = {
            "Executive Overview": "📊",
            "Demand Forecasting": "🔮",
            "Inventory Intelligence": "📦",
            "Product Clustering": "🧩",
            "Customer Segmentation": "👥",
            "Anomaly Detection": "🚨",
            "Model Performance": "🔬",
            "Data Explorer": "🔍"
        }

        selected_page = st.radio(
            "Go to Page",
            pages,
            format_func=lambda p: f"{page_icons[p]}  {p}",
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Pipeline Actions")

        if is_synthetic and st.button("🔄 Reset to Default Dataset", use_container_width=True):
            st.session_state["active_df"] = get_clean_default_data()
            st.session_state["dataset_source"] = "Default Synthetic Retail Dataset"
            st.rerun()

        if st.button("⚡ Retrain ML Models", use_container_width=True):
            retrain_all_models(active_df)
            st.rerun()

        st.markdown("---")
        st.caption(
            "RetailIQ Enterprise v2.4\n"
            "Powered by Scikit-Learn & Streamlit"
        )

    # Top Synthetic Data Disclaimer Notice
    if is_synthetic:
        st.info(
            "ℹ️ **Dataset Notice:** You are viewing the **Synthetic Retail Benchmark Dataset** "
            "(36,500 daily records generated with realistic seasonal patterns, promotions, and lead times). "
            "To analyze real store data, navigate to the **Data Explorer** page to upload your custom CSV.",
            icon="ℹ️"
        )

    # Page Routing
    if selected_page == "Executive Overview":
        render_overview_page(active_df, inventory_plan, eda_summary)
    elif selected_page == "Demand Forecasting":
        render_forecasting_page(active_df, models.get("demand"))
    elif selected_page == "Inventory Intelligence":
        render_inventory_page(active_df, models.get("stockout"))
    elif selected_page == "Product Clustering":
        render_product_clusters_page(active_df, models.get("prod_cluster"))
    elif selected_page == "Customer Segmentation":
        render_customer_segments_page(active_df, models.get("cust_cluster"))
    elif selected_page == "Anomaly Detection":
        render_anomalies_page(active_df, models.get("anomaly"))
    elif selected_page == "Model Performance":
        eval_summary = get_all_models_evaluation_summary()
        render_model_performance_page(eval_summary)
    elif selected_page == "Data Explorer":
        render_data_explorer_page(active_df)


if __name__ == "__main__":
    main()
