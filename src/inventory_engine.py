"""
RetailIQ - Inventory Intelligence Engine
Computes standard supply-chain metrics (Reorder Point, Safety Stock, EOQ, Stock Turnover,
Profit Margin, Inventory Value) and generates multi-factor prescriptive inventory actions
by combining statistical inventory formulas with machine learning predictions.
"""

import numpy as np
import pandas as pd


def compute_safety_stock(
    demand_std: float,
    lead_time_days: int,
    service_level_z: float = 1.65  # 95% service level
) -> int:
    """
    Computes statistical safety stock buffer:
    Safety Stock = Z * sqrt(Lead Time) * Demand Standard Deviation
    """
    std = max(0.2, demand_std)
    lt = max(1, lead_time_days)
    safety = service_level_z * np.sqrt(lt) * std
    return int(np.ceil(safety))


def compute_reorder_point(
    avg_daily_demand: float,
    lead_time_days: int,
    safety_stock: int
) -> int:
    """
    Computes inventory reorder point (ROP):
    Reorder Point = (Average Daily Demand * Supplier Lead Time) + Safety Stock
    """
    lead_time_demand = avg_daily_demand * lead_time_days
    return int(np.ceil(lead_time_demand + safety_stock))


def compute_economic_order_quantity(
    annual_demand: float,
    ordering_cost: float = 45.0,  # Estimated fixed administrative order cost
    holding_cost_unit: float = 5.0
) -> int:
    """
    Computes standard Economic Order Quantity (EOQ):
    EOQ = sqrt((2 * Annual Demand * Ordering Cost) / Holding Cost)
    """
    d = max(1.0, annual_demand)
    s = max(1.0, ordering_cost)
    h = max(0.5, holding_cost_unit)
    eoq = np.sqrt((2.0 * d * s) / h)
    return max(5, int(np.ceil(eoq)))


def evaluate_product_inventory_status(
    df: pd.DataFrame,
    stockout_model_bundle: dict = None
) -> pd.DataFrame:
    """
    Analyzes all products in the retail catalog:
    - Calculates Reorder Point, Safety Stock, Stock Turnover, Profit Margin, Current Value
    - Couples statistical inventory rules with ML Stockout Risk classification
    - Generates actionable, prescriptive inventory decisions
    """
    data = df.copy()
    data["Sales Date"] = pd.to_datetime(data["Sales Date"])

    # Aggregate product history
    grouped = data.groupby("Product ID").agg(
        Product_Name=("Product Name", "first"),
        Category=("Category", "first"),
        Supplier=("Supplier", "first"),
        Unit_Price=("Unit Price", "last"),
        Unit_Cost=("Unit Cost", "last"),
        Current_Stock=("Current Stock", "last"),
        Configured_Reorder=("Reorder Level", "last"),
        Lead_Time=("Lead Time", "last"),
        Avg_Daily_Demand=("Units Sold", "mean"),
        Std_Daily_Demand=("Units Sold", "std"),
        Total_Units_Sold=("Units Sold", "sum"),
        Holding_Cost=("Inventory Holding Cost", "last"),
        Avg_Stock_Level=("Current Stock", "mean")
    ).reset_index()

    grouped["Std_Daily_Demand"] = grouped["Std_Daily_Demand"].fillna(0.5)

    # 1. Financial & Inventory Metrics
    grouped["Inventory_Value"] = (grouped["Current_Stock"] * grouped["Unit_Cost"]).round(2)
    grouped["Profit_Margin_Pct"] = (
        (grouped["Unit_Price"] - grouped["Unit_Cost"]) / (grouped["Unit_Price"] + 1e-4) * 100.0
    ).round(2)

    # Stock Turnover = COGS / Avg Inventory Value
    cogs = grouped["Total_Units_Sold"] * grouped["Unit_Cost"]
    avg_inv_val = grouped["Avg_Stock_Level"] * grouped["Unit_Cost"] + 1e-4
    grouped["Stock_Turnover"] = (cogs / avg_inv_val).round(2)

    # Stock cover days = Current Stock / Avg Daily Demand
    grouped["Stock_Cover_Days"] = (
        grouped["Current_Stock"] / (grouped["Avg_Daily_Demand"] + 1e-4)
    ).round(1)

    # 2. Statistical Inventory Calculations
    grouped["Safety_Stock"] = grouped.apply(
        lambda r: compute_safety_stock(r["Std_Daily_Demand"], int(r["Lead_Time"])), axis=1
    )
    grouped["Calculated_Reorder_Point"] = grouped.apply(
        lambda r: compute_reorder_point(r["Avg_Daily_Demand"], int(r["Lead_Time"]), int(r["Safety_Stock"])), axis=1
    )

    # Annualized demand estimate (365 * Avg Daily Demand)
    grouped["EOQ_Recommended"] = grouped.apply(
        lambda r: compute_economic_order_quantity(
            annual_demand=r["Avg_Daily_Demand"] * 365,
            ordering_cost=50.0,
            holding_cost_unit=max(1.0, r["Holding_Cost"])
        ), axis=1
    )

    # 3. ML Risk Classification Integration (if model provided)
    if stockout_model_bundle is not None:
        model = stockout_model_bundle["model"]
        features = stockout_model_bundle["features"]
        classes = stockout_model_bundle["classes"]

        # Build feature frame
        feat_df = pd.DataFrame()
        feat_df["Current Stock"] = grouped["Current_Stock"]
        feat_df["avg_daily_sales"] = grouped["Avg_Daily_Demand"]
        feat_df["demand_variability"] = (grouped["Std_Daily_Demand"] / (grouped["Avg_Daily_Demand"] + 1e-4)).clip(0, 5)
        feat_df["Lead Time"] = grouped["Lead_Time"]
        feat_df["Reorder Level"] = grouped["Calculated_Reorder_Point"]
        feat_df["stock_cover_days"] = grouped["Stock_Cover_Days"]
        feat_df["lead_time_demand"] = grouped["Avg_Daily_Demand"] * grouped["Lead_Time"]
        feat_df["reorder_gap"] = grouped["Current_Stock"] - grouped["Calculated_Reorder_Point"]
        feat_df["lead_time_coverage_ratio"] = grouped["Current_Stock"] / (feat_df["lead_time_demand"] + 1e-4)

        preds = model.predict(feat_df[features])
        probs = model.predict_proba(feat_df[features])

        grouped["ML_Stockout_Risk"] = preds
        high_idx = classes.index("High Risk") if "High Risk" in classes else 0
        grouped["Stockout_Probability"] = np.round(probs[:, high_idx], 3)
    else:
        # Fallback to rule-based risk if model is not loaded yet
        def rule_risk(row):
            ratio = row["Stock_Cover_Days"] / max(1, row["Lead_Time"])
            if ratio < 0.8:
                return "High Risk"
            elif ratio < 1.4:
                return "Medium Risk"
            return "Low Risk"
        grouped["ML_Stockout_Risk"] = grouped.apply(rule_risk, axis=1)
        grouped["Stockout_Probability"] = grouped["ML_Stockout_Risk"].map(
            {"High Risk": 0.85, "Medium Risk": 0.45, "Low Risk": 0.10}
        )

    # 4. Prescriptive Action Recommendation Engine
    def generate_recommendation(row):
        stock = row["Current_Stock"]
        rop = row["Calculated_Reorder_Point"]
        cover = row["Stock_Cover_Days"]
        turnover = row["Stock_Turnover"]
        margin = row["Profit_Margin_Pct"]
        ml_risk = row["ML_Stockout_Risk"]
        eoq = row["EOQ_Recommended"]

        actions = []
        action_type = "Normal"

        if stock <= rop or ml_risk == "High Risk":
            suggested_qty = max(eoq, int((rop * 1.5) - stock))
            action_type = "Urgent Action Required"
            actions.append(f"Reorder Immediately! (Recommended batch: {suggested_qty:,} units)")
            if ml_risk == "High Risk":
                actions.append(f"Potential stockout imminent within {max(1, int(cover))} days")

        elif stock <= (rop * 1.25) or ml_risk == "Medium Risk":
            action_type = "Warning"
            actions.append("Monitor inventory closely; approaching safety buffer threshold")

        elif cover > 55 and turnover < 35:
            action_type = "Excess Capital"
            actions.append(f"Overstock detected ({int(cover)} days of supply holding ${row['Inventory_Value']:,.0f})")
            if margin > 35:
                actions.append("Consider promotional pricing or bundle discount to accelerate velocity")

        if turnover < 25 and row["Avg_Daily_Demand"] < 3.0:
            actions.append("Slow-moving product; review catalog placement")

        if turnover > 75 and row["Avg_Daily_Demand"] > 8.0:
            actions.append("High-demand top seller; ensure supplier lead-time compliance")

        if not actions:
            actions.append("Inventory balanced; optimal stock coverage")

        primary_recommendation = actions[0]
        details = " | ".join(actions)

        # Suggested Reorder Quantity calculation
        reorder_qty = max(0, int(rop + eoq - stock)) if (stock <= rop or ml_risk == "High Risk") else 0

        return pd.Series([action_type, primary_recommendation, details, reorder_qty])

    rec_cols = ["Action_Type", "Primary_Recommendation", "Detailed_Action_Plan", "Suggested_Reorder_Qty"]
    grouped[rec_cols] = grouped.apply(generate_recommendation, axis=1)

    return grouped.sort_values(by=["Stockout_Probability", "Stock_Cover_Days"], ascending=[False, True]).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import clean_retail_data
    from src.stockout_prediction import load_stockout_model

    raw = load_dataset()
    clean = clean_retail_data(raw)
    try:
        so_bundle = load_stockout_model()
    except Exception:
        so_bundle = None

    inventory_plan = evaluate_product_inventory_status(clean, so_bundle)
    print("Inventory Evaluation Top 5:\n", inventory_plan[[
        "Product ID", "Product_Name", "Current_Stock", "Calculated_Reorder_Point",
        "Stock_Cover_Days", "ML_Stockout_Risk", "Primary_Recommendation", "Suggested_Reorder_Qty"
    ]].head())
