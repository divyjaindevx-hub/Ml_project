# RetailIQ: Comprehensive Technical & Business Architectural Guide

**RetailIQ – Smart Inventory Intelligence & Predictive Analytics Platform**  
*Senior Machine Learning Engineering & Data Science Technical Documentation*

---

## 1. Executive Summary & Business Problem

In enterprise and omnichannel retail, inventory represents the single largest allocation of working capital. Retailers face a continuous, high-stakes trade-off known as the **Inventory Optimization Dilemma**:

```
           UNDER-STOCKING                             OVER-STOCKING
┌───────────────────────────────────┐     ┌───────────────────────────────────┐
│ • Stockouts & Lost Revenue        │     │ • Stagnant Working Capital        │
│ • Churned Customers               │ VS  │ • Escalating Holding Costs (18%)  │
│ • Emergency Expedited Freight     │     │ • Heavy Markdown & Clearance Loss │
│ • Damaged Brand Trust             │     │ • Depreciation & Obsolescence     │
└───────────────────────────────────┘     └───────────────────────────────────┘
                                      ▲
                                      │
                      [ RETAILIQ INTELLIGENCE ENGINE ]
                 Predictive Demand • Automated Replenishment
```

Traditional retail inventory management relies on static reorder thresholds, simple spreadsheet moving averages, or periodic review cycles that fail to capture promotional elasticity, day-of-week seasonality, supplier lead-time volatility, and shifting customer behavior.

**RetailIQ** solves this dilemma by coupling modern supervised Machine Learning (Random Forests for continuous demand regression and discrete stockout risk classification), unsupervised clustering (multidimensional product taxonomies and customer RFM personas), unsupervised anomaly detection (Isolation Forests), and prescriptive supply chain operations research (Reorder Point, Safety Stock, EOQ).

---

## 2. End-to-End System Architecture

RetailIQ is designed as a decoupled, modular system separating data processing, feature engineering, model training, evaluation, supply-chain math, and SaaS dashboard rendering:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           1. DATA INGESTION & PIPELINE                        │
│  • Synthetic Generator (36,500 records, 730 days, 50 SKUs, 350 Customers)     │
│  • Custom CSV Uploader with Automated Schema Validation & Type Coercion       │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                        2. FEATURE ENGINEERING ENGINE                          │
│  • Time-Series Lags (7, 14, 30-day shifted to prevent look-ahead bias)        │
│  • Cyclical Seasonal Fourier Transforms: sin/cos(2π·day/365.25)               │
│  • Inventory Cover Ratios & Lead-Time Demand Multipliers                      │
│  • RFM Customer Aggregation (Recency, Frequency, Monetary, AOV, Discounts)    │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          3. MACHINE LEARNING CORE                             │
│  ┌────────────────────────┐ ┌────────────────────────┐ ┌───────────────────┐ │
│  │ Demand Forecasting     │ │ Stockout Risk          │ │ Product & Customer│ │
│  │ Random Forest Regressor│ │ Random Forest Classif. │ │ K-Means + PCA     │ │
│  └────────────────────────┘ └────────────────────────┘ └───────────────────┘ │
│  ┌───────────────────────────────────────────────────┐                       │
│  │ Anomaly Detection: Isolation Forest (Contam=0.035) │                       │
│  └───────────────────────────────────────────────────┘                       │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                   4. PRESCRIPTIVE INVENTORY ENGINE                            │
│  • Safety Stock (SS = Z · √(Lead Time) · σ_demand)                            │
│  • Reorder Point (ROP = Daily Demand · Lead Time + SS)                        │
│  • Economic Order Quantity (EOQ = √[(2·D·S)/H])                               │
│  • Stock Turnover (COGS / Avg Inventory Value) & Profit Margin Analysis       │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                   5. STREAMLIT SAAS INTERACTIVE DASHBOARD                     │
│  8 Production Pages • Plotly Visualizations • Live What-If Scenario Simulator │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep-Dive: Machine Learning Models & Algorithms

### A. Random Forest Regressor – Demand Forecasting (`src/demand_forecasting.py`)

#### 1. Why Random Forest Regressor?
Standard classical time series models (such as ARIMA/SARIMA) require individual univariate model fitting per SKU, struggle to incorporate external tabular regressors (e.g., promotional discounts, current inventory, supplier lead times), and assume linearity or stationarity. Deep learning approaches (LSTM, Transformers) demand massive compute and lack interpretability.

The **Random Forest Regressor** excels by:
- Capturing non-linear promotional elasticity and weekend/holiday surges.
- Scaling across multiple SKUs using pooled categorical embeddings (`OneHotEncoder`).
- Providing direct feature importance scores for retail governance.
- Maintaining robustness against extreme outliers without gradient explosion.

#### 2. Input Features & Pipeline
```python
NUMERIC_FEATURES = [
    "lag_7", "lag_14", "lag_30",           # Historical sales volumes
    "rolling_mean_7", "rolling_std_7",     # Short-term sales momentum & volatility
    "rolling_mean_30",                     # Long-term baseline demand
    "day_of_week", "month", "is_weekend",  # Calendar features
    "sin_day_of_year", "cos_day_of_year",  # Continuous cyclical annual seasonality
    "Discount", "Unit Price",              # Pricing elasticity
    "Current Stock", "stock_to_reorder_ratio" # Inventory availability
]
CATEGORICAL_FEATURES = ["Category"]
```

#### 3. Strict Prevention of Data Leakage
> [!IMPORTANT]
> A critical failure mode in time-series forecasting is **temporal data leakage**. RetailIQ eliminates leakage through two structural safeguards:
> 1. **Time-Aware Split:** The dataset is partitioned chronologically (80% train: *Jan 2023 – Aug 2024*, 20% test: *Aug 2024 – Dec 2024*). Random $k$-fold cross-validation across time is strictly prohibited.
> 2. **Shifted Rolling Windows:** All rolling statistics (`rolling_mean_7`, `rolling_std_7`, `rolling_mean_30`) are computed after applying `.shift(1)`, guaranteeing that the current day's target sales volume is never present in the feature matrix.

#### 4. Recursive Multi-Step Forecasting
To forecast $n$-days ahead (e.g., next 7 or 30 days), RetailIQ executes recursive auto-regressive projection:
1. Day $t+1$ demand is predicted using historical lags.
2. The predicted value is appended to the simulated history.
3. Lags ($t-7$, $t-30$) and rolling averages are recalculated.
4. Simulated available stock is decremented: $\text{Stock}_{t+1} = \max(0, \text{Stock}_t - \hat{y}_{t+1})$.
5. The model repeats iteratively for all $n$ horizons.

#### 5. Evaluation Metrics
- **Mean Absolute Error (MAE):** $2.558$ units/day
- **Root Mean Squared Error (RMSE):** $3.777$ units/day
- **Test Coefficient of Determination ($R^2$):** $0.6884$ (demonstrating strong predictive fidelity on unseen future periods).

---

### B. Random Forest Classifier – Stockout Risk Prediction (`src/stockout_prediction.py`)

#### 1. Purpose & Risk Stratification
Predicts the likelihood of an inventory stockout event across three calibrated risk tiers:
- **Low Risk:** Inventory buffer exceeds $1.5\times$ lead-time demand; stock is well above reorder levels.
- **Medium Risk:** Inventory cover is between $0.75\times$ and $1.5\times$ lead-time demand; reorder threshold reached.
- **High Risk:** Inventory cover $< 0.75\times$ lead-time demand or warehouse stock $< 35\%$ of reorder point; stockout imminent.

#### 2. Domain-Engineered Ratios
- **Stock Cover Days:** $\text{Cover} = \frac{\text{Current Stock}}{\text{Average Daily Demand} + \epsilon}$
- **Lead-Time Demand:** $\text{LTD} = \text{Average Daily Demand} \times \text{Supplier Lead Time}$
- **Lead-Time Coverage Ratio:** $\text{Ratio} = \frac{\text{Current Stock}}{\text{LTD} + \epsilon}$
- **Reorder Gap:** $\text{Gap} = \text{Current Stock} - \text{Reorder Level}$

#### 3. Performance Metrics
- **Overall Accuracy:** $100\%$ on stratified holdout test split.
- **Macro Precision, Recall, and F1 Score:** $1.000$
- **Multiclass One-vs-Rest ROC-AUC:** $1.000$

---

### C. K-Means Clustering – Product Performance Taxonomy (`src/product_clustering.py`)

#### 1. Methodology
Unsupervised multidimensional product clustering eliminates arbitrary manual SKU tiering (such as naive ABC classification) by simultaneously analyzing:
1. `Avg_Daily_Sales`: Volume velocity
2. `Total_Revenue`: Financial contribution
3. `Profit_Margin_Pct`: Percentage margin
4. `Stock_Turnover`: Velocity of capital deployment
5. `Demand_Variability`: Volatility ($\sigma / \mu$)
6. `Current_Stock`: Warehouse footprint

#### 2. Normalization & Diagnostics
- **StandardScaler:** Normalizes features to zero mean and unit variance ($\mu = 0, \sigma = 1$).
- **Elbow Method:** Evaluates sum of squared distances (inertia) for $k \in [2, 8]$.
- **Silhouette Analysis:** Maximizes inter-cluster distance while minimizing intra-cluster dispersion (Optimal $k=4$, Silhouette Score = $0.3336$).
- **PCA 2D Mapping:** Principal Component Analysis compresses 6 dimensions into 2 orthogonal axes explaining $\approx 55\%$ of catalog variance for 2D visual inspection.

#### 3. Automated Data-Driven Cluster Personas (No Hardcoded Labels)
RetailIQ dynamically interprets clusters based on their calculated relative averages:
- **Fast-Moving & High-Velocity:** Above-average sales and high inventory turnover ($> 80\times$). Requires tight supplier lead times and strict stock replenishment.
- **High-Margin Cash Cows:** Above-median revenue with strong profit margins ($> 50\%$). Essential to maintain high service availability.
- **Slow-Moving / Overstock Risk:** Below-average sales combined with high warehouse inventory levels. Candidates for promotional markdown or bundle deals.
- **Core Steady Performers:** Balanced velocity, predictable demand, and stable inventory holding patterns.

---

### D. K-Means Clustering – Customer Behavioral RFM Segmentation (`src/customer_segmentation.py`)

#### 1. Behavioral Features
Extracts customer transaction patterns:
- **Recency ($R$):** Days elapsed between customer's most recent purchase and the dataset reference date.
- **Frequency ($F$):** Total number of distinct completed transactions.
- **Monetary ($M$):** Aggregate lifetime spending (\$).
- **Average Order Value (AOV):** $\frac{\text{Monetary}}{\text{Frequency}}$.
- **Average Discount Sensitivity:** Percentage of purchases involving promotional markdown.

#### 2. Segment Archetypes & Actionable CRM Strategies
- **High-Value Champions:** High spend, high frequency, recent purchases. Strategy: VIP loyalty rewards, exclusive early access, dedicated account management.
- **Discount & Promotion Seekers:** High discount responsiveness ($> 15\%$). Strategy: Targeted clearance promotions, flash sales, price-drop alerts.
- **At-Risk / Lapsed Buyers:** High recency ($> 30$ days dormant), low frequency. Strategy: Automated win-back campaigns, feedback surveys, personalized re-engagement vouchers.
- **Steady Occasional Shoppers:** Moderate frequency and spend. Strategy: Cross-sell recommendations, bundle discounts, free-shipping thresholds.

---

### E. Isolation Forest – Anomaly Detection (`src/anomaly_detection.py`)

#### 1. Algorithmic Principle
Anomalous events (flash spikes, catastrophic stockouts, supply disruptions) are structurally "few and different". The **Isolation Forest** builds an ensemble of $130$ isolation trees by randomly selecting features and split values. Anomalous observations require significantly fewer splits to isolate (shorter path lengths from the tree root).

#### 2. Anomaly Scoring & Contamination
- Contamination parameter: $\alpha = 0.035$ ($3.5\%$ expected anomaly rate).
- Raw decision scores are transformed into a normalized $[0, 1]$ severity score where values $> 0.65$ denote acute operational deviations.

#### 3. Natural-Language Explainability Module
Unlike "black box" anomaly flags, RetailIQ evaluates which feature caused the isolation and produces transparent explanations:
- `Unusual Demand Spike (+7.7σ above baseline)`: Sudden surge in sales units compared to the product's 30-day mean.
- `Critical Stockout Depletion (Zero Available Inventory)`: Zero stock combined with unmet customer transactions.
- `Failed Promotion (High Discount 20% with Sluggish Demand)`: Heavy promotional discount failing to stimulate sales lift.
- `Rapid Stock Depletion`: Excessive stock drained in a single inventory cycle.

---

## 4. Prescriptive Inventory Intelligence Engine

The Inventory Intelligence Engine (`src/inventory_engine.py`) translates statistical models into concrete procurement decisions:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PROCUREMENT DECISION MATRIX                     │
├────────────────────────────────┬───────────────────────────────────────┤
│ Condition                      │ Prescriptive Recommendation           │
├────────────────────────────────┼───────────────────────────────────────┤
│ Stock ≤ Reorder Point (ROP)    │ 🚨 Reorder Immediately!               │
│ OR ML Risk == "High Risk"      │    Suggested Qty = max(EOQ, ROP - Stk)│
├────────────────────────────────┼───────────────────────────────────────┤
│ Stock ≤ 1.25 × ROP             │ ⚠️ Monitor Inventory Closely          │
│ OR ML Risk == "Medium Risk"    │    Approaching safety buffer threshold│
├────────────────────────────────┼───────────────────────────────────────┤
│ Stock Cover > 55 Days          │ 📦 Overstock Detected                 │
│ AND Turnover < 35x             │    Holding excess stagnant capital    │
├────────────────────────────────┼───────────────────────────────────────┤
│ Overstock Detected             │ 🏷️ Consider Promotional Pricing       │
│ AND Profit Margin > 35%        │    Discount to recover working capital│
├────────────────────────────────┼───────────────────────────────────────┤
│ Turnover < 25x                 │ ⏳ Slow-Moving Product                │
│ AND Daily Demand < 3.0 units   │    Review catalog placement or delist │
├────────────────────────────────┼───────────────────────────────────────┤
│ Turnover > 75x                 │ 🏆 High-Demand Top Seller             │
│ AND Daily Demand > 8.0 units   │    Enforce strict supplier lead time  │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 5. Interactive Dashboard Architecture (8 Pages)

```
PAGE 1: EXECUTIVE OVERVIEW
├── 6 High-Level KPI Cards (SKUs, Valuation, Revenue, Units, Margin, High Risk)
├── Revenue & Units Sales Momentum (Daily / Weekly / Monthly resample toggle)
├── Category Revenue Share Donut Chart
├── Inventory Valuation Breakdown by Category
├── Top 8 Selling Products by Revenue
├── Monthly Financial Performance (Revenue vs. Gross Profit)
└── Stock Health Gauge (Low, Medium, High Risk distribution)

PAGE 2: DEMAND FORECASTING
├── Product Selector Dropdown (50 SKUs)
├── Historical Sales vs. ML Model Validation Holdout
├── 7-Day Weekly Recursive Forecast Horizon (Bar chart + metrics)
├── 30-Day Monthly Recursive Forecast Horizon (Sales vs. Stock runout)
├── Day-of-Week & Monthly Seasonality Patterns
└── Interactive What-If Scenario Simulator (Custom price, discount, stock)

PAGE 3: INVENTORY INTELLIGENCE
├── High-Level Procurement KPIs (Immediate reorders, overstocked SKUs)
├── Quadrant Scatter: Stock Turnover vs. Profit Margin Matrix
├── Supplier Lead-Time Reliability & Risk Exposure Analysis
└── Filterable Inventory Table with Category/Supplier/Risk filters & CSV export

PAGE 4: PRODUCT CLUSTERING
├── 2D PCA Dimensionality Reduction Scatter Map
├── Cluster Proportions Bar Chart
├── Elbow Method Curve (Inertia vs. k) & Silhouette Score Curve
├── Quantitative Cluster Behavioral Profiling Table
└── Individual SKU Cluster Assignment Table with Search

PAGE 5: CUSTOMER SEGMENTATION
├── Customer Spend vs. Purchase Frequency Scatter Plot
├── 2D PCA Customer Projection
├── Customer Segment Proportions Donut Chart
├── Average Order Value (AOV) by Segment
└── Prescriptive CRM Retention Playbook Table

PAGE 6: ANOMALY DETECTION
├── Inliers vs. Anomalies Sales Timeline (Filterable by Product)
├── Category Anomaly Distribution Bar Chart
├── Anomaly Severity Score Histogram
└── Anomaly Audit Register with Natural-Language Root Cause Explanations

PAGE 7: MODEL PERFORMANCE
├── Demand Regressor: MAE, RMSE, R², Train vs. Test, Feature Importance
├── Stockout Classifier: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
├── Clustering Diagnostics: Elbow Curves & Silhouette Coefficients
└── Isolation Forest Diagnostics: Contamination rate & feature explanations

PAGE 8: DATA EXPLORER
├── Filterable Dataset Browser with Multi-Category Filter
├── Custom CSV Uploader with Automated Schema & Column Verification
├── Pearson Correlation Heatmap for Numeric Features
├── Missing Value & Null Percentage Audit Table
└── One-Click Filtered Dataset CSV Export
```

---

## 6. Execution & Troubleshooting Guide

### Running Locally
```bash
# 1. Navigate to directory
cd "/Users/divyjain/Desktop/ipl predictor"

# 2. Run Streamlit application
python3 -m streamlit run app.py
```
Open browser: `http://localhost:8501`

### Retraining Models from CLI
```bash
PYTHONPATH=. python3 src/data_loader.py
PYTHONPATH=. python3 src/preprocessing.py
PYTHONPATH=. python3 src/demand_forecasting.py
PYTHONPATH=. python3 src/stockout_prediction.py
PYTHONPATH=. python3 src/product_clustering.py
PYTHONPATH=. python3 src/customer_segmentation.py
PYTHONPATH=. python3 src/anomaly_detection.py
```

### Running the Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 7. Business Impact & ROI

1. **Working Capital Recovery:** Pinpoints slow-moving and overstocked SKUs, enabling markdown campaigns before products depreciate.
2. **Stockout Prevention:** Proactively flags at-risk items $7$ to $21$ days before runout, safeguarding sales revenue.
3. **Data-Driven Replenishment:** Replaces arbitrary gut-feeling orders with mathematically optimal EOQ and Reorder Points.
4. **CRM Revenue Optimization:** Matches promotion-sensitive customers with excess inventory while preserving margins for VIP buyers.
