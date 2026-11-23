"""
RetailIQ - Data Loader & Synthetic Data Generation Module
Handles creation of realistic retail transaction and inventory datasets,
CSV loading, schema validation, and synthetic data identification.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_RAW_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DEFAULT_DATA_FILE = os.path.join(DATA_RAW_PATH, "retail_inventory.csv")

# Standard required columns for full RetailIQ functionality
REQUIRED_COLUMNS = [
    "Product ID",
    "Product Name",
    "Category",
    "Supplier",
    "Unit Price",
    "Unit Cost",
    "Current Stock",
    "Reorder Level",
    "Units Sold",
    "Sales Date",
    "Discount",
    "Customer ID",
    "Customer Segment",
    "Lead Time",
    "Stockout Indicator",
    "Inventory Holding Cost"
]

CATEGORIES = [
    "Electronics",
    "Home & Kitchen",
    "Apparel & Footwear",
    "Health & Wellness",
    "Sports & Outdoors"
]

SUPPLIERS = [
    "TechCorp Global",
    "Apex Logistics",
    "Horizon Goods",
    "Nexus Supply Co",
    "Prime Source Distribution"
]

PRODUCTS_SPEC = [
    # Electronics
    {"id": "PRD-001", "name": "Wireless Noise-Canceling Headphones", "cat": "Electronics", "base_price": 149.99, "cost_ratio": 0.55, "lead": 7, "velocity": "fast", "supplier": "TechCorp Global"},
    {"id": "PRD-002", "name": "Smart Fitness Watch Ultra", "cat": "Electronics", "base_price": 199.99, "cost_ratio": 0.52, "lead": 10, "velocity": "medium", "supplier": "TechCorp Global"},
    {"id": "PRD-003", "name": "4K Ultra HD Gaming Monitor 27-inch", "cat": "Electronics", "base_price": 349.99, "cost_ratio": 0.65, "lead": 14, "velocity": "slow", "supplier": "TechCorp Global"},
    {"id": "PRD-004", "name": "Mechanical RGB Keyboard", "cat": "Electronics", "base_price": 89.99, "cost_ratio": 0.50, "lead": 5, "velocity": "fast", "supplier": "Nexus Supply Co"},
    {"id": "PRD-005", "name": "Portable Bluetooth Speaker 20W", "cat": "Electronics", "base_price": 59.99, "cost_ratio": 0.48, "lead": 6, "velocity": "fast", "supplier": "TechCorp Global"},
    {"id": "PRD-006", "name": "USB-C Multiport Docking Station", "cat": "Electronics", "base_price": 69.99, "cost_ratio": 0.50, "lead": 8, "velocity": "medium", "supplier": "Nexus Supply Co"},
    {"id": "PRD-007", "name": "Wireless Charging Pad Duo", "cat": "Electronics", "base_price": 34.99, "cost_ratio": 0.45, "lead": 4, "velocity": "fast", "supplier": "Nexus Supply Co"},
    {"id": "PRD-008", "name": "Ergonomic Vertical Mouse", "cat": "Electronics", "base_price": 44.99, "cost_ratio": 0.45, "lead": 5, "velocity": "medium", "supplier": "TechCorp Global"},
    {"id": "PRD-009", "name": "Smart Wi-Fi Security Camera", "cat": "Electronics", "base_price": 79.99, "cost_ratio": 0.55, "lead": 12, "velocity": "medium", "supplier": "Apex Logistics"},
    {"id": "PRD-010", "name": "True Wireless Earbuds Pro", "cat": "Electronics", "base_price": 119.99, "cost_ratio": 0.50, "lead": 7, "velocity": "fast", "supplier": "TechCorp Global"},

    # Home & Kitchen
    {"id": "PRD-011", "name": "Stainless Steel Cookware Set 10-pc", "cat": "Home & Kitchen", "base_price": 229.99, "cost_ratio": 0.58, "lead": 15, "velocity": "slow", "supplier": "Horizon Goods"},
    {"id": "PRD-012", "name": "Programmable Drip Coffee Maker", "cat": "Home & Kitchen", "base_price": 89.99, "cost_ratio": 0.52, "lead": 9, "velocity": "medium", "supplier": "Horizon Goods"},
    {"id": "PRD-013", "name": "Digital Air Fryer XL 6-Quart", "cat": "Home & Kitchen", "base_price": 119.99, "cost_ratio": 0.54, "lead": 10, "velocity": "fast", "supplier": "Apex Logistics"},
    {"id": "PRD-014", "name": "High-Speed Countertop Blender", "cat": "Home & Kitchen", "base_price": 99.99, "cost_ratio": 0.50, "lead": 8, "velocity": "medium", "supplier": "Horizon Goods"},
    {"id": "PRD-015", "name": "Cast Iron Dutch Oven 5-Qt", "cat": "Home & Kitchen", "base_price": 69.99, "cost_ratio": 0.50, "lead": 12, "velocity": "slow", "supplier": "Horizon Goods"},
    {"id": "PRD-016", "name": "Ergonomic Memory Foam Pillow", "cat": "Home & Kitchen", "base_price": 39.99, "cost_ratio": 0.40, "lead": 5, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-017", "name": "Electric Gooseneck Kettle", "cat": "Home & Kitchen", "base_price": 49.99, "cost_ratio": 0.45, "lead": 7, "velocity": "medium", "supplier": "Horizon Goods"},
    {"id": "PRD-018", "name": "Glass Food Storage Set 18-pc", "cat": "Home & Kitchen", "base_price": 34.99, "cost_ratio": 0.42, "lead": 6, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-019", "name": "Robotic Vacuum Cleaner Slim", "cat": "Home & Kitchen", "base_price": 279.99, "cost_ratio": 0.62, "lead": 16, "velocity": "slow", "supplier": "Apex Logistics"},
    {"id": "PRD-020", "name": "Stainless Steel Chef Knife 8-in", "cat": "Home & Kitchen", "base_price": 45.99, "cost_ratio": 0.45, "lead": 6, "velocity": "medium", "supplier": "Horizon Goods"},

    # Apparel & Footwear
    {"id": "PRD-021", "name": "Men's Performance Running Shoes", "cat": "Apparel & Footwear", "base_price": 129.99, "cost_ratio": 0.48, "lead": 11, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-022", "name": "Women's Lightweight Trail Runners", "cat": "Apparel & Footwear", "base_price": 119.99, "cost_ratio": 0.48, "lead": 10, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-023", "name": "Breathable Athletic Hoodie", "cat": "Apparel & Footwear", "base_price": 64.99, "cost_ratio": 0.42, "lead": 8, "velocity": "medium", "supplier": "Prime Source Distribution"},
    {"id": "PRD-024", "name": "Moisture-Wicking Training Shorts", "cat": "Apparel & Footwear", "base_price": 29.99, "cost_ratio": 0.38, "lead": 5, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-025", "name": "Waterproof Outdoor Shell Jacket", "cat": "Apparel & Footwear", "base_price": 179.99, "cost_ratio": 0.52, "lead": 14, "velocity": "slow", "supplier": "Prime Source Distribution"},
    {"id": "PRD-026", "name": "Thermal Compression Tights", "cat": "Apparel & Footwear", "base_price": 44.99, "cost_ratio": 0.40, "lead": 6, "velocity": "medium", "supplier": "Prime Source Distribution"},
    {"id": "PRD-027", "name": "Merino Wool Hiking Socks 3-Pack", "cat": "Apparel & Footwear", "base_price": 24.99, "cost_ratio": 0.35, "lead": 4, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-028", "name": "Lightweight Puffer Vest", "cat": "Apparel & Footwear", "base_price": 84.99, "cost_ratio": 0.46, "lead": 9, "velocity": "slow", "supplier": "Horizon Goods"},
    {"id": "PRD-029", "name": "Canvas Travel Duffle Bag 40L", "cat": "Apparel & Footwear", "base_price": 74.99, "cost_ratio": 0.45, "lead": 7, "velocity": "medium", "supplier": "Apex Logistics"},
    {"id": "PRD-030", "name": "Polarized Sports Sunglasses", "cat": "Apparel & Footwear", "base_price": 54.99, "cost_ratio": 0.38, "lead": 5, "velocity": "fast", "supplier": "Nexus Supply Co"},

    # Health & Wellness
    {"id": "PRD-031", "name": "Sonic Electric Toothbrush Elite", "cat": "Health & Wellness", "base_price": 79.99, "cost_ratio": 0.45, "lead": 7, "velocity": "fast", "supplier": "Nexus Supply Co"},
    {"id": "PRD-032", "name": "Deep Tissue Percussion Massager", "cat": "Health & Wellness", "base_price": 139.99, "cost_ratio": 0.50, "lead": 11, "velocity": "medium", "supplier": "Apex Logistics"},
    {"id": "PRD-033", "name": "Organic Whey Protein Powder 2lb", "cat": "Health & Wellness", "base_price": 42.99, "cost_ratio": 0.50, "lead": 5, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-034", "name": "Plant-Based Multivitamin 90-ct", "cat": "Health & Wellness", "base_price": 28.99, "cost_ratio": 0.35, "lead": 4, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-035", "name": "Aromatherapy Ultrasonic Diffuser", "cat": "Health & Wellness", "base_price": 32.99, "cost_ratio": 0.40, "lead": 5, "velocity": "medium", "supplier": "Horizon Goods"},
    {"id": "PRD-036", "name": "Bluetooth Smart Body Scale", "cat": "Health & Wellness", "base_price": 49.99, "cost_ratio": 0.46, "lead": 8, "velocity": "medium", "supplier": "TechCorp Global"},
    {"id": "PRD-037", "name": "Blue Light Blocking Glasses", "cat": "Health & Wellness", "base_price": 22.99, "cost_ratio": 0.30, "lead": 3, "velocity": "fast", "supplier": "Nexus Supply Co"},
    {"id": "PRD-038", "name": "Weighted Sleep Blanket 15lb", "cat": "Health & Wellness", "base_price": 89.99, "cost_ratio": 0.52, "lead": 12, "velocity": "slow", "supplier": "Horizon Goods"},
    {"id": "PRD-039", "name": "Collagen Peptides Powder 16oz", "cat": "Health & Wellness", "base_price": 36.99, "cost_ratio": 0.48, "lead": 5, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-040", "name": "Infrared Forehead Thermometer", "cat": "Health & Wellness", "base_price": 26.99, "cost_ratio": 0.38, "lead": 4, "velocity": "medium", "supplier": "Nexus Supply Co"},

    # Sports & Outdoors
    {"id": "PRD-041", "name": "Premium High-Density Yoga Mat", "cat": "Sports & Outdoors", "base_price": 39.99, "cost_ratio": 0.42, "lead": 6, "velocity": "fast", "supplier": "Horizon Goods"},
    {"id": "PRD-042", "name": "Adjustable Dumbbell Set 50lb", "cat": "Sports & Outdoors", "base_price": 299.99, "cost_ratio": 0.60, "lead": 18, "velocity": "slow", "supplier": "Apex Logistics"},
    {"id": "PRD-043", "name": "Insulated Stainless Bottle 32oz", "cat": "Sports & Outdoors", "base_price": 27.99, "cost_ratio": 0.35, "lead": 4, "velocity": "fast", "supplier": "Prime Source Distribution"},
    {"id": "PRD-044", "name": "Resistance Loop Bands Set of 5", "cat": "Sports & Outdoors", "base_price": 18.99, "cost_ratio": 0.30, "lead": 3, "velocity": "fast", "supplier": "Nexus Supply Co"},
    {"id": "PRD-045", "name": "Camping Backpacking Tent 2-Person", "cat": "Sports & Outdoors", "base_price": 159.99, "cost_ratio": 0.55, "lead": 14, "velocity": "slow", "supplier": "Apex Logistics"},
    {"id": "PRD-046", "name": "Hydration Pack Backpack 2L", "cat": "Sports & Outdoors", "base_price": 46.99, "cost_ratio": 0.44, "lead": 7, "velocity": "medium", "supplier": "Horizon Goods"},
    {"id": "PRD-047", "name": "Trekking Poles Carbon Fiber Pair", "cat": "Sports & Outdoors", "base_price": 58.99, "cost_ratio": 0.46, "lead": 8, "velocity": "medium", "supplier": "Apex Logistics"},
    {"id": "PRD-048", "name": "Speed Jump Rope Aluminum Handle", "cat": "Sports & Outdoors", "base_price": 16.99, "cost_ratio": 0.32, "lead": 3, "velocity": "fast", "supplier": "Nexus Supply Co"},
    {"id": "PRD-049", "name": "LED Rechargeable Headlamp 500lm", "cat": "Sports & Outdoors", "base_price": 24.99, "cost_ratio": 0.35, "lead": 4, "velocity": "fast", "supplier": "Nexus Supply Co"},
    {"id": "PRD-050", "name": "Inflatable Stand Up Paddleboard", "cat": "Sports & Outdoors", "base_price": 389.99, "cost_ratio": 0.62, "lead": 20, "velocity": "slow", "supplier": "Apex Logistics"}
]

CUSTOMER_SEGMENT_DIST = {
    "Frequent Shopper": 0.30,
    "Budget Conscious": 0.35,
    "Premium Buyer": 0.20,
    "Occasional": 0.15
}


def generate_synthetic_retail_data(
    num_days: int = 730,
    start_date: str = "2023-01-01",
    output_path: str = DEFAULT_DATA_FILE,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic synthetic retail inventory and transaction records
    spanning historical dates with seasonal patterns, promotions, lead times,
    stockout risks, customer purchases, and injected realistic anomalies.
    """
    np.random.seed(random_seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [base_date + timedelta(days=i) for i in range(num_days)]

    # Generate customer pool
    num_customers = 350
    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    customer_segments = np.random.choice(
        list(CUSTOMER_SEGMENT_DIST.keys()),
        size=num_customers,
        p=list(CUSTOMER_SEGMENT_DIST.values())
    )
    customer_map = dict(zip(customer_ids, customer_segments))

    records = []

    # Simulation state per product
    # Track inventory levels over time for realism
    product_state = {}
    for p in PRODUCTS_SPEC:
        cost = round(p["base_price"] * p["cost_ratio"], 2)
        base_daily_demand = 12 if p["velocity"] == "fast" else (6 if p["velocity"] == "medium" else 2)
        # Initial stock: ~ 3x to 5x lead time demand
        initial_stock = int(base_daily_demand * p["lead"] * np.random.uniform(2.5, 4.5))
        reorder_point = int(base_daily_demand * p["lead"] + 1.65 * np.sqrt(p["lead"]) * (base_daily_demand * 0.3))
        holding_cost = round(cost * 0.18, 2)  # 18% annual holding cost per unit

        product_state[p["id"]] = {
            "current_stock": initial_stock,
            "reorder_level": reorder_point,
            "unit_cost": cost,
            "base_demand": base_daily_demand,
            "holding_cost": holding_cost,
            "pending_restock_days": -1,
            "pending_restock_qty": 0
        }

    for day_idx, current_date in enumerate(dates):
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        month = current_date.month
        day_of_year = current_date.timetuple().tm_yday

        # Macro seasonal factor (Q4 holiday surge + summer peak)
        holiday_factor = 1.0 + 0.35 * np.sin(2 * np.pi * (day_of_year - 40) / 365) + (0.3 if month in [11, 12] else 0.0)
        weekend_factor = 1.35 if day_of_week in [5, 6] else 1.0

        for p in PRODUCTS_SPEC:
            pid = p["id"]
            state = product_state[pid]

            # Check if pending order arrives today
            if state["pending_restock_days"] == 0:
                state["current_stock"] += state["pending_restock_qty"]
                state["pending_restock_days"] = -1
                state["pending_restock_qty"] = 0
            elif state["pending_restock_days"] > 0:
                state["pending_restock_days"] -= 1

            # Category-specific seasonal nuance
            cat_multiplier = 1.0
            if p["cat"] == "Sports & Outdoors" and month in [5, 6, 7, 8]:
                cat_multiplier = 1.4
            elif p["cat"] == "Apparel & Footwear" and month in [9, 10, 11]:
                cat_multiplier = 1.3
            elif p["cat"] == "Electronics" and month in [11, 12]:
                cat_multiplier = 1.55

            # Promotional discount probability (more likely on weekends/holidays)
            has_discount = np.random.rand() < (0.28 if day_of_week in [4, 5, 6] or month in [11, 12] else 0.12)
            discount = round(np.random.choice([0.05, 0.10, 0.15, 0.20, 0.25]), 2) if has_discount else 0.0
            discount_lift = 1.0 + (discount * 2.5)  # 20% discount gives ~ 1.5x demand

            expected_demand = state["base_demand"] * holiday_factor * weekend_factor * cat_multiplier * discount_lift
            # Poisson or negative binomial variation
            sales_units = max(0, int(np.random.poisson(lam=max(0.8, expected_demand))))

            # Injected anomalies: occasional demand spike (flash sale) or stock drop
            is_anomaly = False
            if np.random.rand() < 0.004:  # ~0.4% chance of notable anomaly
                sales_units = int(sales_units * np.random.uniform(3.5, 5.0))
                is_anomaly = True

            # Stockout check
            current_stock = state["current_stock"]
            stockout_flag = 1 if current_stock == 0 else 0

            # Realized sales cannot exceed stock if strict inventory, or can trigger stockout
            actual_sold = min(sales_units, current_stock) if current_stock > 0 else 0
            if actual_sold < sales_units and current_stock > 0:
                stockout_flag = 1

            state["current_stock"] = max(0, current_stock - actual_sold)

            # Reorder trigger check
            if state["current_stock"] <= state["reorder_level"] and state["pending_restock_days"] == -1:
                # Trigger order
                lead_time = p["lead"] + np.random.choice([-1, 0, 1, 2], p=[0.15, 0.60, 0.20, 0.05])
                lead_time = max(2, lead_time)
                # Reorder quantity: Economic Order Quantity approximation
                restock_order = int(state["base_demand"] * p["lead"] * 3.5)
                state["pending_restock_days"] = lead_time
                state["pending_restock_qty"] = restock_order

            # Sample customer
            cust_idx = np.random.randint(0, num_customers)
            cust_id = customer_ids[cust_idx]
            cust_seg = customer_map[cust_id]

            # Price adjustment if discounted
            effective_price = round(p["base_price"] * (1.0 - discount), 2)

            records.append({
                "Product ID": pid,
                "Product Name": p["name"],
                "Category": p["cat"],
                "Supplier": p["supplier"],
                "Unit Price": effective_price,
                "Unit Cost": state["unit_cost"],
                "Current Stock": state["current_stock"],
                "Reorder Level": state["reorder_level"],
                "Units Sold": actual_sold,
                "Sales Date": current_date.strftime("%Y-%m-%d"),
                "Discount": discount,
                "Customer ID": cust_id,
                "Customer Segment": cust_seg,
                "Lead Time": p["lead"],
                "Stockout Indicator": stockout_flag,
                "Inventory Holding Cost": state["holding_cost"],
                "Is_Anomaly_Injected": 1 if is_anomaly else 0,
                "Is_Synthetic": 1
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"[DATA GENERATOR] Generated {len(df):,} synthetic retail transaction records.")
    print(f"[DATA GENERATOR] Saved to {output_path}")
    return df


def load_dataset(filepath: str = DEFAULT_DATA_FILE) -> pd.DataFrame:
    """
    Loads dataset from CSV, performs schema validation, type conversions,
    and returns a clean pandas DataFrame. If file doesn't exist, generates it.
    """
    if not os.path.exists(filepath):
        print(f"[DATA LOADER] File not found at '{filepath}'. Generating default synthetic dataset...")
        return generate_synthetic_retail_data(output_path=filepath)

    df = pd.read_csv(filepath)

    # Standardize column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Parse dates
    if "Sales Date" in df.columns:
        df["Sales Date"] = pd.to_datetime(df["Sales Date"], errors="coerce")
        # Drop invalid dates if any
        df = df.dropna(subset=["Sales Date"])

    # Ensure numeric columns are properly typed
    numeric_cols = [
        "Unit Price", "Unit Cost", "Current Stock", "Reorder Level",
        "Units Sold", "Discount", "Lead Time", "Stockout Indicator",
        "Inventory Holding Cost"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Handle duplicates if any
    df = df.drop_duplicates()

    return df


def validate_uploaded_data(df: pd.DataFrame) -> dict:
    """
    Validates user-uploaded CSV dataset against expected schema,
    checks missing values, returns validation status, warnings, and summary stats.
    """
    status = {"is_valid": True, "errors": [], "warnings": [], "info": {}}

    if df is None or df.empty:
        status["is_valid"] = False
        status["errors"].append("The uploaded dataset is empty.")
        return status

    # Check for missing required columns
    present_cols = set(df.columns.str.strip())
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in present_cols]

    if missing_cols:
        status["warnings"].append(f"Missing recommended columns: {', '.join(missing_cols)}")
        # Check critical columns needed for basic inventory and forecasting
        critical_cols = ["Product ID", "Sales Date", "Units Sold"]
        critical_missing = [c for c in critical_cols if c not in present_cols]
        if critical_missing:
            status["is_valid"] = False
            status["errors"].append(f"Critical columns missing: {', '.join(critical_missing)}")

    # Check missing values
    null_counts = df.isnull().sum()
    high_null_cols = null_counts[null_counts > 0].to_dict()
    if high_null_cols:
        status["warnings"].append(f"Columns with null values detected: {high_null_cols}")

    # General stats
    status["info"]["total_rows"] = len(df)
    status["info"]["total_columns"] = len(df.columns)
    if "Product ID" in df.columns:
        status["info"]["unique_products"] = df["Product ID"].nunique()
    if "Sales Date" in df.columns:
        status["info"]["date_min"] = str(df["Sales Date"].min())
        status["info"]["date_max"] = str(df["Sales Date"].max())

    return status


if __name__ == "__main__":
    df = generate_synthetic_retail_data(num_days=730)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(3))
