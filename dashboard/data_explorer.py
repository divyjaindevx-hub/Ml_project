"""
RetailIQ Dashboard - Page 8: Data Explorer & Dataset Management
Enables CSV upload, automated schema validation, missing value inspection,
correlation matrix heatmaps, interactive filtering, and data export.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import validate_uploaded_data
from src.preprocessing import clean_retail_data


def render_data_explorer_page(df: pd.DataFrame):
    st.markdown("## 🔍 Data Explorer & Dataset Management")
    st.markdown(
        "Upload custom retail CSV datasets, inspect schema compatibility, audit missing data, "
        "explore correlation matrices, and export cleaned datasets."
    )

    t1, t2, t3 = st.tabs(["1. Dataset Preview & Filters", "2. Upload Custom CSV", "3. Correlation Analysis & Missing Values"])

    with t1:
        st.subheader("📋 Dataset Browser")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            cats = df["Category"].unique().tolist() if "Category" in df.columns else []
            selected_cats = st.multiselect("Filter Categories", cats, default=cats, key="de_cat")
        with col_f2:
            search = st.text_input("Search by Product Name or ID", "", key="de_search")

        filtered = df.copy()
        if selected_cats and "Category" in filtered.columns:
            filtered = filtered[filtered["Category"].isin(selected_cats)]
        if search:
            mask = filtered["Product ID"].astype(str).str.contains(search, case=False)
            if "Product Name" in filtered.columns:
                mask = mask | filtered["Product Name"].astype(str).str.contains(search, case=False)
            filtered = filtered[mask]

        st.caption(f"Showing {len(filtered):,} matching records of {len(df):,} total.")
        st.dataframe(filtered.head(100), use_container_width=True)

        # Download clean data
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Dataset (CSV)",
            data=csv_bytes,
            file_name="RetailIQ_Filtered_Data.csv",
            mime="text/csv"
        )

    with t2:
        st.subheader("📤 Upload Custom Enterprise CSV")
        st.markdown(
            "Upload your store or warehouse transaction CSV. RetailIQ will automatically validate the columns, "
            "report missing fields, clean data types, and permit live analysis."
        )

        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_uploaded = pd.read_csv(uploaded_file)
                st.success(f"Successfully uploaded `{uploaded_file.name}` with {len(raw_uploaded):,} rows.")

                # Validation
                val_res = validate_uploaded_data(raw_uploaded)
                if val_res["is_valid"]:
                    st.success("✅ Dataset schema is valid and ready for pipeline processing!")
                else:
                    st.error("❌ Dataset failed schema validation:")
                    for err in val_res["errors"]:
                        st.write(f"- {err}")

                if val_res["warnings"]:
                    st.warning("⚠️ Schema Warnings / Missing Non-Critical Columns:")
                    for w in val_res["warnings"]:
                        st.write(f"- {w}")

                st.markdown("##### Uploaded Data Preview:")
                st.dataframe(raw_uploaded.head(10), use_container_width=True)

                if st.button("🚀 Load Uploaded Data into Active Session"):
                    cleaned_upload = clean_retail_data(raw_uploaded)
                    st.session_state["active_df"] = cleaned_upload
                    st.session_state["dataset_source"] = f"Custom Upload: {uploaded_file.name}"
                    st.success("Active session dataset updated! Navigate through dashboard pages to view results.")

            except Exception as e:
                st.error(f"Error reading uploaded CSV: {e}")

    with t3:
        st.subheader("🔗 Correlation Heatmap & Missing Value Audit")

        c_col1, c_col2 = st.columns([7, 5])

        with c_col1:
            st.markdown("##### Pearson Correlation Matrix")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            # Remove IDs or binary synthetic flags from correlation
            exclude_cols = ["Is_Synthetic", "Is_Anomaly_Injected", "Stockout Indicator"]
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                fig_corr.update_layout(
                    margin=dict(l=10, r=10, t=25, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=380
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Insufficient numeric columns for correlation heatmap.")

        with c_col2:
            st.markdown("##### Missing Value Audit")
            null_counts = df.isnull().sum().reset_index()
            null_counts.columns = ["Column", "Null Count"]
            null_counts["Null Percentage"] = (null_counts["Null Count"] / len(df) * 100.0).round(2)
            st.dataframe(null_counts, use_container_width=True, hide_index=True)
