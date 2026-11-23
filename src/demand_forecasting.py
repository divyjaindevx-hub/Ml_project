"""
RetailIQ - Demand Forecasting Module
Implements Random Forest Regressor for product demand forecasting,
featuring time-aware train/test splitting, lag-feature pipeline,
recursive multi-step future forecasting, and scenario prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.feature_engineering import build_daily_product_demand_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEMAND_MODEL_FILE = os.path.join(MODELS_DIR, "demand_model.pkl")

NUMERIC_FEATURES = [
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_30",
    "day_of_week",
    "month",
    "is_weekend",
    "sin_day_of_year",
    "cos_day_of_year",
    "Discount",
    "Unit Price",
    "Current Stock",
    "stock_to_reorder_ratio"
]

CATEGORICAL_FEATURES = ["Category"]


def train_demand_model(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
    save_path: str = DEMAND_MODEL_FILE,
    random_state: int = 42
) -> dict:
    """
    Trains a time-aware Random Forest Regressor for demand forecasting.
    Uses chronological cutoff for train/test split to prevent temporal data leakage.
    Returns model artifact bundle including pipeline, metrics, importances, and evaluation data.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[DEMAND FORECASTING] Engineering time-series features...")
    featured_df = build_daily_product_demand_features(df)

    # Time-aware train/test split based on Sales Date cutoff
    unique_dates = np.sort(featured_df["Sales Date"].unique())
    split_idx = int(len(unique_dates) * (1.0 - test_ratio))
    split_date = unique_dates[split_idx]

    train_mask = featured_df["Sales Date"] < split_date
    test_mask = featured_df["Sales Date"] >= split_date

    train_data = featured_df[train_mask].copy()
    test_data = featured_df[test_mask].copy()

    X_train = train_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_data["Units Sold"].values

    X_test = test_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_data["Units Sold"].values

    print(f"[DEMAND FORECASTING] Train rows: {len(X_train):,}, Test rows: {len(X_test):,}")
    print(f"[DEMAND FORECASTING] Train period: {train_data['Sales Date'].min().date()} to {train_data['Sales Date'].max().date()}")
    print(f"[DEMAND FORECASTING] Test period:  {test_data['Sales Date'].min().date()} to {test_data['Sales Date'].max().date()}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    print("[DEMAND FORECASTING] Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Compute metrics
    metrics = {
        "train_mae": round(float(mean_absolute_error(y_train, y_train_pred)), 3),
        "train_rmse": round(float(root_mean_squared_error(y_train, y_train_pred)), 3),
        "train_r2": round(float(r2_score(y_train, y_train_pred)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, y_test_pred)), 3),
        "test_rmse": round(float(root_mean_squared_error(y_test, y_test_pred)), 3),
        "test_r2": round(float(r2_score(y_test, y_test_pred)), 4)
    }

    print(f"[DEMAND FORECASTING] Results: Test MAE={metrics['test_mae']}, Test RMSE={metrics['test_rmse']}, Test R²={metrics['test_r2']}")

    # Feature importances
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERIC_FEATURES + cat_feature_names
    importances = pipeline.named_steps["regressor"].feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Prepare evaluation dataframe with actuals and predictions
    test_eval_df = test_data[["Sales Date", "Product ID", "Product Name", "Category", "Units Sold"]].copy()
    test_eval_df["Predicted_Demand"] = np.round(y_test_pred, 1)
    test_eval_df["Error"] = np.round(test_eval_df["Units Sold"] - test_eval_df["Predicted_Demand"], 1)

    model_bundle = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_importances": feature_importance_df,
        "feature_names": all_feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_eval_df": test_eval_df,
        "split_date": str(split_date),
        "featured_df": featured_df
    }

    joblib.dump(model_bundle, save_path)
    print(f"[DEMAND FORECASTING] Saved model bundle to {save_path}")

    return model_bundle


def load_demand_model(model_path: str = DEMAND_MODEL_FILE) -> dict:
    """Loads saved demand forecasting model bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)


def forecast_future_demand(
    product_id: str,
    n_days: int = 30,
    model_bundle: dict = None,
    df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generates multi-step recursive demand forecasts for the next n_days for a specific product.
    Updates rolling lag features iteratively to project future daily sales.
    """
    if model_bundle is None:
        model_bundle = load_demand_model()

    pipeline = model_bundle["pipeline"]
    featured_df = model_bundle.get("featured_df")

    if featured_df is None and df is not None:
        featured_df = build_daily_product_demand_features(df)

    prod_df = featured_df[featured_df["Product ID"] == product_id].sort_values("Sales Date").copy()
    if prod_df.empty:
        raise ValueError(f"Product ID '{product_id}' not found in dataset.")

    latest_record = prod_df.iloc[-1].copy()
    last_date = latest_record["Sales Date"]
    cat = latest_record["Category"]
    unit_price = latest_record["Unit Price"]
    current_stock = latest_record["Current Stock"]
    reorder_level = latest_record["Reorder Level"]

    # History of sales units for rolling calculations
    sales_history = prod_df["Units Sold"].values.tolist()

    forecasts = []
    simulated_stock = current_stock

    for i in range(1, n_days + 1):
        future_date = last_date + timedelta(days=i)
        day_of_week = future_date.weekday()
        month = future_date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0
        day_of_year = future_date.timetuple().tm_yday
        sin_doy = np.sin(2 * np.pi * day_of_year / 365.25)
        cos_doy = np.cos(2 * np.pi * day_of_year / 365.25)

        # Lags from history
        lag_7 = sales_history[-7] if len(sales_history) >= 7 else sales_history[-1]
        lag_14 = sales_history[-14] if len(sales_history) >= 14 else sales_history[-1]
        lag_30 = sales_history[-30] if len(sales_history) >= 30 else sales_history[-1]

        rolling_7 = np.mean(sales_history[-7:])
        rolling_std_7 = np.std(sales_history[-7:]) if len(sales_history) >= 7 else 0.5
        rolling_30 = np.mean(sales_history[-30:])

        stock_ratio = min(10.0, simulated_stock / (reorder_level + 1e-4))

        input_row = pd.DataFrame([{
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_30": lag_30,
            "rolling_mean_7": rolling_7,
            "rolling_std_7": rolling_std_7,
            "rolling_mean_30": rolling_30,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "sin_day_of_year": sin_doy,
            "cos_day_of_year": cos_doy,
            "Discount": 0.05 if is_weekend else 0.0,
            "Unit Price": unit_price,
            "Current Stock": simulated_stock,
            "stock_to_reorder_ratio": stock_ratio,
            "Category": cat
        }])

        pred_sales = float(pipeline.predict(input_row)[0])
        pred_sales = max(0.0, round(pred_sales, 1))

        # Append to forecast
        forecasts.append({
            "Forecast Date": future_date.strftime("%Y-%m-%d"),
            "Product ID": product_id,
            "Product Name": latest_record["Product Name"],
            "Category": cat,
            "Predicted Demand": pred_sales,
            "Simulated Available Stock": max(0, int(simulated_stock))
        })

        # Update history for next recursive step
        sales_history.append(pred_sales)
        simulated_stock = max(0, simulated_stock - pred_sales)

    return pd.DataFrame(forecasts)


def predict_demand_scenario(
    category: str,
    unit_price: float,
    current_stock: int,
    recent_sales_mean: float,
    expected_discount: float = 0.0,
    model_bundle: dict = None
) -> float:
    """
    Scenario prediction function for arbitrary user inputs.
    Uses the trained pipeline to estimate expected daily sales.
    """
    if model_bundle is None:
        model_bundle = load_demand_model()

    pipeline = model_bundle["pipeline"]

    # Construct representative scenario features
    reorder_est = max(10, int(recent_sales_mean * 7))
    stock_ratio = min(10.0, current_stock / (reorder_est + 1e-4))

    input_df = pd.DataFrame([{
        "lag_7": recent_sales_mean,
        "lag_14": recent_sales_mean,
        "lag_30": recent_sales_mean,
        "rolling_mean_7": recent_sales_mean,
        "rolling_std_7": max(0.5, recent_sales_mean * 0.25),
        "rolling_mean_30": recent_sales_mean,
        "day_of_week": 3,  # Mid-week baseline
        "month": 6,        # Mid-year baseline
        "is_weekend": 0,
        "sin_day_of_year": 0.0,
        "cos_day_of_year": 1.0,
        "Discount": expected_discount,
        "Unit Price": unit_price,
        "Current Stock": current_stock,
        "stock_to_reorder_ratio": stock_ratio,
        "Category": category
    }])

    prediction = pipeline.predict(input_df)[0]
    return max(0.0, round(float(prediction), 1))


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import clean_retail_data

    raw = load_dataset()
    clean = clean_retail_data(raw)
    bundle = train_demand_model(clean)

    sample_forecast = forecast_future_demand("PRD-001", n_days=7, model_bundle=bundle)
    print("Sample 7-Day Forecast for PRD-001:\n", sample_forecast)
