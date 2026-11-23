"""
RetailIQ Dashboard - Page 7: Model Performance & Algorithmic Diagnostics
Presents rigorous evaluation metrics for all four machine learning models:
Demand Regression (MAE, RMSE, R²), Stockout Classification (Accuracy, Precision,
Recall, F1, Confusion Matrix), K-Means Clustering (Silhouette, Inertia),
and Isolation Forest (Contamination, Anomaly Breakdown) plus Feature Importances.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.model_evaluation import get_all_models_evaluation_summary


def render_model_performance_page(eval_summary: dict = None):
    st.markdown("## 🔬 Machine Learning Model Evaluation & Governance")
    st.markdown(
        "Inspect algorithmic performance metrics, loss functions, feature importances, "
        "confusion matrices, and cross-validation integrity across all trained production models."
    )

    if eval_summary is None:
        eval_summary = get_all_models_evaluation_summary()

    m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
        "1. Demand Forecasting (Regression)",
        "2. Stockout Risk (Classification)",
        "3. Product & Customer Clustering",
        "4. Anomaly Detection (Isolation Forest)"
    ])

    # TAB 1: Demand Regression
    with m_tab1:
        st.subheader("🌲 Random Forest Regressor – Demand Forecasting")
        st.markdown(
            "Trained with **time-aware chronological validation** on 28,000 training records and 7,000 holdout test records. "
            "Evaluates future daily demand without data leakage."
        )

        demand_info = eval_summary.get("demand", {})
        if demand_info.get("status") == "Trained":
            metrics = demand_info["metrics"]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Test MAE", f"{metrics['test_mae']} units")
            with c2:
                st.metric("Test RMSE", f"{metrics['test_rmse']} units")
            with c3:
                st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
            with c4:
                st.metric("Train R² Score", f"{metrics['train_r2']:.4f}")

            st.markdown("---")
            col_d1, col_d2 = st.columns([7, 5])

            with col_d1:
                st.markdown("##### 🔑 Feature Importance Drivers")
                fi_df = pd.DataFrame(demand_info["feature_importances"]).sort_values("importance", ascending=True).tail(12)

                fig_fi = px.bar(
                    fi_df, x="importance", y="feature",
                    orientation="h",
                    labels={"importance": "Gini Impurity Importance", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Blues"
                )
                fig_fi.update_layout(
                    margin=dict(l=10, r=10, t=25, b=10),
                    coloraxis_showscale=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            with col_d2:
                st.markdown("##### 📋 Regression Performance Diagnostic")
                perf_df = pd.DataFrame({
                    "Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Coefficient of Determination (R²)"],
                    "Train Set": [metrics["train_mae"], metrics["train_rmse"], metrics["train_r2"]],
                    "Test Set": [metrics["test_mae"], metrics["test_rmse"], metrics["test_r2"]]
                })
                st.dataframe(perf_df, use_container_width=True, hide_index=True)

                st.info(
                    "💡 **Interpretability Note:** Rolling 7-day sales and 30-day moving averages represent the dominant "
                    "predictive weights, reflecting realistic retail momentum and consumer re-order frequency."
                )
        else:
            st.warning("Demand model metrics are not available.")

    # TAB 2: Stockout Classification
    with m_tab2:
        st.subheader("🎯 Random Forest Classifier – Stockout Risk Prediction")
        st.markdown(
            "Multi-class stockout risk classifier categorizing inventory into **Low Risk**, **Medium Risk**, and **High Risk** "
            "with balanced class weighting."
        )

        stockout_info = eval_summary.get("stockout", {})
        if stockout_info.get("status") == "Trained":
            s_metrics = stockout_info["metrics"]

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Overall Accuracy", f"{s_metrics['accuracy']*100:.2f}%")
            with s2:
                st.metric("Macro Precision", f"{s_metrics['precision_macro']*100:.2f}%")
            with s3:
                st.metric("Macro Recall", f"{s_metrics['recall_macro']*100:.2f}%")
            with s4:
                st.metric("Multiclass ROC-AUC", f"{s_metrics['roc_auc']:.4f}")

            st.markdown("---")
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                st.markdown("##### 🔲 Confusion Matrix Heatmap")
                classes = s_metrics["classes"]
                cm = np.array(s_metrics["confusion_matrix"])

                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    x=classes,
                    y=classes,
                    text_auto=True,
                    color_continuous_scale="Teal"
                )
                fig_cm.update_layout(
                    margin=dict(l=10, r=10, t=25, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=340
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_s2:
                st.markdown("##### 🔑 Stockout Risk Feature Importance")
                so_fi = pd.DataFrame(stockout_info["feature_importances"]).sort_values("importance", ascending=True)

                fig_so_fi = px.bar(
                    so_fi, x="importance", y="feature",
                    orientation="h",
                    labels={"importance": "Importance", "feature": ""},
                    color="importance",
                    color_continuous_scale="Reds"
                )
                fig_so_fi.update_layout(
                    margin=dict(l=10, r=10, t=25, b=10),
                    coloraxis_showscale=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=340
                )
                st.plotly_chart(fig_so_fi, use_container_width=True)
        else:
            st.warning("Stockout model metrics are not available.")

    # TAB 3: Clustering Diagnostics
    with m_tab3:
        st.subheader("🧩 K-Means Clustering Validation")
        col_c1, col_c2 = st.columns(2)

        p_cluster = eval_summary.get("product_clustering", {})
        c_cluster = eval_summary.get("customer_segmentation", {})

        with col_c1:
            st.markdown("##### 📦 Product Segmentation Metrics")
            if p_cluster.get("status") == "Trained":
                st.metric("Optimal Clusters (k)", p_cluster["n_clusters"])
                st.metric("Silhouette Coefficient", f"{p_cluster['silhouette_score']:.4f}")
                st.metric("PCA Variance (2D)", f"{(p_cluster['explained_variance'][0] + p_cluster['explained_variance'][1])*100:.1f}%")

                fig_p_elb = px.line(
                    x=p_cluster["k_values"], y=p_cluster["elbow_inertias"],
                    markers=True,
                    labels={"x": "k (Number of Clusters)", "y": "Inertia"}
                )
                fig_p_elb.update_traces(line_color="#0284C7")
                fig_p_elb.update_layout(
                    title="Product Elbow Curve",
                    margin=dict(l=10, r=10, t=35, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=240
                )
                st.plotly_chart(fig_p_elb, use_container_width=True)

        with col_c2:
            st.markdown("##### 👥 Customer RFM Clustering Metrics")
            if c_cluster.get("status") == "Trained":
                st.metric("Customer Segments (k)", c_cluster["n_clusters"])
                st.metric("Silhouette Coefficient", f"{c_cluster['silhouette_score']:.4f}")
                st.metric("PCA Variance (2D)", f"{(c_cluster['explained_variance'][0] + c_cluster['explained_variance'][1])*100:.1f}%")

                fig_c_elb = px.line(
                    x=c_cluster["k_values"], y=c_cluster["elbow_inertias"],
                    markers=True,
                    labels={"x": "k (Number of Clusters)", "y": "Inertia"}
                )
                fig_c_elb.update_traces(line_color="#10B981")
                fig_c_elb.update_layout(
                    title="Customer Elbow Curve",
                    margin=dict(l=10, r=10, t=35, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=240
                )
                st.plotly_chart(fig_c_elb, use_container_width=True)

    # TAB 4: Anomaly Detection
    with m_tab4:
        st.subheader("🚨 Isolation Forest Diagnostics")
        anom_info = eval_summary.get("anomaly_detection", {})
        if anom_info.get("status") == "Trained":
            stats = anom_info["summary_stats"]

            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.metric("Contamination Prior", f"{stats['contamination']*100:.1f}%")
            with a2:
                st.metric("Total Inliers (Normal)", f"{stats['normal_count']:,}")
            with a3:
                st.metric("Total Anomalies Flagged", f"{stats['anomaly_count']:,}")
            with a4:
                st.metric("Actual Flagged Percentage", f"{stats['anomaly_pct']:.2f}%")

            st.markdown("---")
            st.markdown("##### 📐 Features Used in Anomaly Detection Tree Ensemble:")
            feat_list = stats["features_used"]
            for f in feat_list:
                st.markdown(f"- **`{f}`**: Evaluates extreme tail deviations from product baseline behaviour.")
        else:
            st.warning("Anomaly model metrics not available.")
