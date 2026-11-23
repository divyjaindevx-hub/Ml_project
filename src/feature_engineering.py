"""
RetailIQ - Feature Engineering Module
Builds time-aware features for demand forecasting, inventory ratios
for stockout risk classification, RFM profiles for customer segmentation,
and aggregate performance profiles for product clustering.
"""

import numpy as np
import pandas as pd


def build_daily_product_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds time-series features for demand forecasting on daily product sales:
    - Lag features (lag 7, 14, 30)
    - Rolling window statistics (7-day, 30-day mean and std) strictly shifted to avoid leakage
    - Calendar and cyclical seasonal features (day of week, month, sine/cosine day of year)
    - Price, discount, stock availability, and categorical indicators
    """
    data = df.copy()
    data["Sales Date"] = pd.to_datetime(data["Sales Date"])

    # Aggregate to daily product level in case data contains multiple customer transactions per day
    agg_funcs = {
        "Units Sold": "sum",
        "Unit Price": "mean",
        "Unit Cost": "mean",
        "Discount": "mean",
        "Current Stock": "last",
        "Reorder Level": "last",
        "Lead Time": "last",
        "Stockout Indicator": "max"
    }

    group_cols = ["Product ID"]
    extra_cols = ["Product Name", "Category", "Supplier"]
    for ec in extra_cols:
        if ec in data.columns:
            group_cols.append(ec)

    daily = data.groupby(group_cols + [pd.Grouper(key="Sales Date", freq="D")]).agg(agg_funcs).reset_index()

    # Sort strictly by Product ID and Sales Date
    daily = daily.sort_values(by=["Product ID", "Sales Date"]).reset_index(drop=True)

    # Feature engineering per product with shift to prevent future data leakage
    features_list = []
    for pid, group in daily.groupby("Product ID"):
        g = group.sort_values("Sales Date").copy()

        # Lag features
        g["lag_7"] = g["Units Sold"].shift(7)
        g["lag_14"] = g["Units Sold"].shift(14)
        g["lag_30"] = g["Units Sold"].shift(30)

        # Rolling statistics (shift by 1 so current day sales are NOT included)
        g["rolling_mean_7"] = g["Units Sold"].shift(1).rolling(window=7, min_periods=3).mean()
        g["rolling_std_7"] = g["Units Sold"].shift(1).rolling(window=7, min_periods=3).std().fillna(0)
        g["rolling_mean_30"] = g["Units Sold"].shift(1).rolling(window=30, min_periods=7).mean()

        # Price and inventory dynamics
        g["stock_to_reorder_ratio"] = (g["Current Stock"] / (g["Reorder Level"] + 1e-4)).clip(0, 10)

        features_list.append(g)

    featured_df = pd.concat(features_list, ignore_index=True)

    # Calendar and seasonal features
    featured_df["day_of_week"] = featured_df["Sales Date"].dt.dayofweek
    featured_df["month"] = featured_df["Sales Date"].dt.month
    featured_df["quarter"] = featured_df["Sales Date"].dt.quarter
    featured_df["is_weekend"] = featured_df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical seasonal transformation for day of year
    day_of_year = featured_df["Sales Date"].dt.dayofyear
    featured_df["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    featured_df["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Drop early warm-up records where lag_30 is NaN to ensure high-quality training data
    clean_featured = featured_df.dropna(subset=["lag_7", "lag_30", "rolling_mean_7"]).reset_index(drop=True)

    return clean_featured


def build_stockout_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features for Stockout Risk Classification:
    - Current inventory
    - Average daily demand (over 30-day window or product lifetime)
    - Demand variability (std / mean)
    - Supplier lead time
    - Reorder level
    - Days of inventory remaining (Stock Cover)
    - Lead time demand ratio
    - Target: Stockout_Risk (Low Risk, Medium Risk, High Risk)
    """
    data = df.copy()
    data["Sales Date"] = pd.to_datetime(data["Sales Date"])

    # Compute product-level average daily sales and variability
    product_stats = data.groupby("Product ID").agg(
        avg_daily_sales=("Units Sold", "mean"),
        std_daily_sales=("Units Sold", "std"),
        unit_price=("Unit Price", "last"),
        unit_cost=("Unit Cost", "last"),
        lead_time=("Lead Time", "last")
    ).reset_index()

    product_stats["std_daily_sales"] = product_stats["std_daily_sales"].fillna(0.5)
    product_stats["demand_variability"] = (
        product_stats["std_daily_sales"] / (product_stats["avg_daily_sales"] + 1e-4)
    ).clip(0, 5)

    # Merge stats back to records (sampling periodic inventory snapshots or latest dates)
    merged = pd.merge(data, product_stats[["Product ID", "avg_daily_sales", "demand_variability"]], on="Product ID", how="left")

    # Domain engineered inventory ratios
    merged["lead_time_demand"] = merged["avg_daily_sales"] * merged["Lead Time"]
    merged["stock_cover_days"] = merged["Current Stock"] / (merged["avg_daily_sales"] + 1e-4)
    merged["reorder_gap"] = merged["Current Stock"] - merged["Reorder Level"]
    merged["lead_time_coverage_ratio"] = merged["Current Stock"] / (merged["lead_time_demand"] + 1e-4)

    # Define ground-truth risk category based on operational inventory science:
    # High Risk: stock cover < 0.75 * lead time or current stock <= 0.3 * reorder level
    # Medium Risk: stock cover between 0.75 * lead time and 1.5 * lead time
    # Low Risk: stock cover >= 1.5 * lead time and current stock > reorder level
    def assign_risk_class(row):
        ratio = row["lead_time_coverage_ratio"]
        stock = row["Current Stock"]
        reorder = row["Reorder Level"]

        if stock == 0 or ratio < 0.75 or stock < (0.35 * reorder):
            return "High Risk"
        elif ratio < 1.4 or stock <= (1.05 * reorder):
            return "Medium Risk"
        else:
            return "Low Risk"

    merged["Stockout_Risk"] = merged.apply(assign_risk_class, axis=1)

    return merged


def build_customer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Customer Recency, Frequency, Monetary (RFM) and behavioral features:
    - Recency: Days since most recent transaction
    - Frequency: Total purchase transactions
    - Monetary: Total spending ($)
    - Average Order Value (AOV)
    - Average discount taken
    - Preferred product category
    """
    data = df.copy()
    data["Sales Date"] = pd.to_datetime(data["Sales Date"])
    ref_date = data["Sales Date"].max()

    if "Revenue" not in data.columns:
        data["Revenue"] = data["Units Sold"] * data["Unit Price"]

    # Filter to transactions where units sold > 0
    tx_data = data[data["Units Sold"] > 0].copy()

    # Customer level aggregation
    rfm = tx_data.groupby("Customer ID").agg(
        last_date=("Sales Date", "max"),
        Frequency=("Sales Date", "count"),
        Monetary=("Revenue", "sum"),
        Total_Units=("Units Sold", "sum"),
        Avg_Discount=("Discount", "mean"),
        Preferred_Category=("Category", lambda x: x.mode().iloc[0] if not x.empty else "General")
    ).reset_index()

    rfm["Recency"] = (ref_date - rfm["last_date"]).dt.days
    rfm["Avg_Order_Value"] = (rfm["Monetary"] / (rfm["Frequency"] + 1e-4)).round(2)
    rfm["Monetary"] = rfm["Monetary"].round(2)

    return rfm


def build_product_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes product-level performance features for K-Means product segmentation:
    - Average daily sales
    - Total revenue
    - Profit margin (%)
    - Stock turnover (COGS / Avg Inventory Value)
    - Demand variability (CV = std / mean)
    - Current stock level
    - Current inventory value
    """
    data = df.copy()
    data["Sales Date"] = pd.to_datetime(data["Sales Date"])

    if "Revenue" not in data.columns:
        data["Revenue"] = data["Units Sold"] * data["Unit Price"]
    if "Profit" not in data.columns:
        data["Profit"] = data["Revenue"] - (data["Units Sold"] * data["Unit Cost"])

    prod_summary = data.groupby("Product ID").agg(
        Product_Name=("Product Name", "first"),
        Category=("Category", "first"),
        Supplier=("Supplier", "first"),
        Avg_Daily_Sales=("Units Sold", "mean"),
        Std_Daily_Sales=("Units Sold", "std"),
        Total_Units_Sold=("Units Sold", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum"),
        Unit_Price=("Unit Price", "last"),
        Unit_Cost=("Unit Cost", "last"),
        Current_Stock=("Current Stock", "last"),
        Reorder_Level=("Reorder Level", "last"),
        Lead_Time=("Lead Time", "last"),
        Avg_Stock=("Current Stock", "mean")
    ).reset_index()

    # Derived KPIs
    prod_summary["Profit_Margin_Pct"] = (
        (prod_summary["Unit_Price"] - prod_summary["Unit_Cost"]) / (prod_summary["Unit_Price"] + 1e-4) * 100.0
    ).round(2)

    # Cost of Goods Sold = Total Units Sold * Unit Cost
    cogs = prod_summary["Total_Units_Sold"] * prod_summary["Unit_Cost"]
    avg_inventory_val = prod_summary["Avg_Stock"] * prod_summary["Unit_Cost"] + 1e-4
    prod_summary["Stock_Turnover"] = (cogs / avg_inventory_val).round(2)

    # Demand variability
    prod_summary["Demand_Variability"] = (
        prod_summary["Std_Daily_Sales"].fillna(0.1) / (prod_summary["Avg_Daily_Sales"] + 1e-4)
    ).clip(0, 5).round(3)

    # Current inventory valuation
    prod_summary["Current_Inventory_Value"] = (prod_summary["Current_Stock"] * prod_summary["Unit_Cost"]).round(2)

    return prod_summary


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import clean_retail_data

    raw = load_dataset()
    clean = clean_retail_data(raw)

    print("Building demand features...")
    df_demand = build_daily_product_demand_features(clean)
    print(f"Demand features shape: {df_demand.shape}")

    print("Building stockout features...")
    df_stockout = build_stockout_classification_features(clean)
    print(f"Stockout features shape: {df_stockout.shape}")
    print("Risk classes:", df_stockout["Stockout_Risk"].value_counts().to_dict())

    print("Building RFM features...")
    df_rfm = build_customer_rfm_features(clean)
    print(f"RFM features shape: {df_rfm.shape}")

    print("Building product cluster features...")
    df_prod = build_product_clustering_features(clean)
    print(f"Product cluster features shape: {df_prod.shape}")
