"""
RetailIQ - Data Cleaning and Preprocessing Module
Cleans raw retail records, handles nulls, normalizes schemas,
formats dates, and computes baseline statistical summaries.
"""

import os
import numpy as np
import pandas as pd

DATA_PROCESSED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
CLEANED_DATA_FILE = os.path.join(DATA_PROCESSED_PATH, "cleaned_inventory.csv")


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw retail dataset:
    - Normalizes column names
    - Converts date strings to datetime
    - Coerces numeric types with zero/median imputation
    - Deduplicates identical records
    - Ensures chronological ordering per product
    """
    if df is None or df.empty:
        raise ValueError("Provided dataframe is empty or None.")

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    # Parse date
    if "Sales Date" in cleaned.columns:
        cleaned["Sales Date"] = pd.to_datetime(cleaned["Sales Date"], errors="coerce")
        cleaned = cleaned.dropna(subset=["Sales Date"])

    # Ensure required numeric types
    num_cols = {
        "Unit Price": 0.0,
        "Unit Cost": 0.0,
        "Current Stock": 0,
        "Reorder Level": 0,
        "Units Sold": 0,
        "Discount": 0.0,
        "Lead Time": 7,
        "Stockout Indicator": 0,
        "Inventory Holding Cost": 0.0
    }

    for col, default_val in num_cols.items():
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(default_val)
            if col in ["Current Stock", "Reorder Level", "Units Sold", "Lead Time", "Stockout Indicator"]:
                cleaned[col] = cleaned[col].astype(int)
            else:
                cleaned[col] = cleaned[col].astype(float)

    # String columns hygiene
    str_cols = ["Product ID", "Product Name", "Category", "Supplier", "Customer ID", "Customer Segment"]
    for col in str_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("Unknown").astype(str).str.strip()

    # Deduplicate
    cleaned = cleaned.drop_duplicates()

    # Compute Revenue column if missing
    if "Revenue" not in cleaned.columns and "Units Sold" in cleaned.columns and "Unit Price" in cleaned.columns:
        cleaned["Revenue"] = (cleaned["Units Sold"] * cleaned["Unit Price"]).round(2)

    # Compute Profit column if missing
    if "Profit" not in cleaned.columns and "Revenue" in cleaned.columns and "Unit Cost" in cleaned.columns:
        cost_of_goods = (cleaned["Units Sold"] * cleaned["Unit Cost"]).round(2)
        cleaned["Profit"] = (cleaned["Revenue"] - cost_of_goods).round(2)

    # Sort chronologically by product
    if "Product ID" in cleaned.columns and "Sales Date" in cleaned.columns:
        cleaned = cleaned.sort_values(by=["Product ID", "Sales Date"]).reset_index(drop=True)

    return cleaned


def calculate_eda_summary(df: pd.DataFrame) -> dict:
    """
    Computes comprehensive exploratory data analysis metrics
    for reporting and executive dashboard overviews.
    """
    if df is None or df.empty:
        return {}

    total_revenue = float(df["Revenue"].sum()) if "Revenue" in df.columns else float((df["Units Sold"] * df["Unit Price"]).sum())
    total_cost = float((df["Units Sold"] * df["Unit Cost"]).sum()) if "Unit Cost" in df.columns else 0.0
    total_profit = total_revenue - total_cost
    avg_margin = (total_profit / total_revenue * 100.0) if total_revenue > 0 else 0.0

    summary = {
        "total_records": len(df),
        "total_products": int(df["Product ID"].nunique()) if "Product ID" in df.columns else 0,
        "total_categories": int(df["Category"].nunique()) if "Category" in df.columns else 0,
        "total_suppliers": int(df["Supplier"].nunique()) if "Supplier" in df.columns else 0,
        "total_customers": int(df["Customer ID"].nunique()) if "Customer ID" in df.columns else 0,
        "total_units_sold": int(df["Units Sold"].sum()) if "Units Sold" in df.columns else 0,
        "total_revenue": round(total_revenue, 2),
        "total_profit": round(total_profit, 2),
        "avg_profit_margin_pct": round(avg_margin, 2),
        "stockout_events": int(df["Stockout Indicator"].sum()) if "Stockout Indicator" in df.columns else 0,
        "stockout_rate_pct": round(float(df["Stockout Indicator"].mean() * 100.0), 2) if "Stockout Indicator" in df.columns else 0.0,
        "date_range_start": str(df["Sales Date"].min().date()) if "Sales Date" in df.columns else "N/A",
        "date_range_end": str(df["Sales Date"].max().date()) if "Sales Date" in df.columns else "N/A"
    }

    # Latest inventory snapshot value
    if "Current Stock" in df.columns and "Unit Cost" in df.columns and "Product ID" in df.columns and "Sales Date" in df.columns:
        latest_date = df["Sales Date"].max()
        latest_df = df[df["Sales Date"] == latest_date]
        summary["current_inventory_units"] = int(latest_df["Current Stock"].sum())
        summary["current_inventory_value"] = round(float((latest_df["Current Stock"] * latest_df["Unit Cost"]).sum()), 2)

    return summary


def save_processed_data(df: pd.DataFrame, output_path: str = CLEANED_DATA_FILE) -> str:
    """Saves cleaned dataset to processed data directory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[PREPROCESSING] Cleaned dataset saved with {len(df):,} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    from src.data_loader import load_dataset
    raw_df = load_dataset()
    clean_df = clean_retail_data(raw_df)
    save_processed_data(clean_df)
    eda = calculate_eda_summary(clean_df)
    print("[PREPROCESSING] EDA Summary:", eda)
