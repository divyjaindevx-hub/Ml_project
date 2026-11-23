# RetailIQ – Smart Inventory Intelligence & Predictive Analytics Platform

**RetailIQ** is an enterprise-grade Machine Learning and Inventory Intelligence application designed to solve critical retail supply chain challenges: demand forecasting, stockout risk mitigation, slow-moving inventory identification, customer RFM segmentation, anomaly detection, and automated inventory replenishment.

---

## 🌟 Key Capabilities

1. **Time-Aware Demand Forecasting (Random Forest Regressor)**:
   - Evaluates multi-horizon sales velocity using non-leaking lag features (7-day, 14-day, 30-day lags, rolling statistics, calendar dynamics, and promotional elasticity).
   - Generates recursive 7-day and 30-day future demand projections with simulated warehouse drawdown.

2. **Stockout Risk Classification (Random Forest Classifier)**:
   - Categorizes catalog SKUs into `Low Risk`, `Medium Risk`, and `High Risk` of stockout with calibrated probabilities.
   - Evaluates stock cover days, lead-time demand, supplier reliability, and historical demand volatility.

3. **Multidimensional Product Taxonomy (K-Means + PCA)**:
   - Partitions products into operational archetypes (*Fast-Moving & High-Velocity*, *High-Margin Cash Cows*, *Slow-Moving / Overstock Risk*, and *Core Steady Performers*) without arbitrary hardcoded thresholds.
   - Includes Elbow curve (inertia) validation, Silhouette score diagnostics, and 2D PCA variance projections.

4. **Customer Behavioral RFM Segmentation (K-Means)**:
   - Segments the customer base using Recency, Frequency, Monetary (RFM), Average Order Value (AOV), and discount sensitivity.
   - Generates prescriptive CRM retention strategies for *High-Value Champions*, *Promotion Seekers*, *At-Risk / Lapsed Buyers*, and *Steady Occasional Shoppers*.

5. **Operational Anomaly Detection (Isolation Forest)**:
   - Flags abnormal sales surges, demand drops, failed promotions, and unexplained stock depletion events.
   - Provides natural-language root-cause explanations for each flagged anomaly.

6. **Prescriptive Inventory Intelligence Engine**:
   - Integrates classical supply-chain science with machine learning outputs:
     - **Reorder Point (ROP):** $\text{ROP} = (\text{Avg Daily Demand} \times \text{Lead Time}) + \text{Safety Stock}$
     - **Safety Stock:** $\text{SS} = Z \times \sqrt{\text{Lead Time}} \times \sigma_{\text{demand}}$
     - **Stock Turnover:** $\text{Turnover} = \frac{\text{COGS}}{\text{Average Inventory Value}}$
     - **Profit Margin:** $\text{Margin} = \frac{\text{Unit Price} - \text{Unit Cost}}{\text{Unit Price}} \times 100$
     - **Economic Order Quantity (EOQ):** $\text{EOQ} = \sqrt{\frac{2 \times \text{Annual Demand} \times \text{Order Cost}}{\text{Holding Cost}}}$

7. **Professional SaaS-Style Interactive Dashboard**:
   - Built with Streamlit and Plotly across 8 specialized pages.
   - Supports live scenario simulation, interactive product filters, CSV uploads, schema validation, and CSV reporting exports.

---

## 📂 Project Architecture

```
RetailIQ/
│
├── app.py                      # Main Streamlit SaaS application entrypoint
├── requirements.txt            # Project dependencies
├── README.md                   # Complete architectural documentation
│
├── data/
│   ├── raw/
│   │   └── retail_inventory.csv     # 36,500 historical transaction & inventory records
│   └── processed/
│       └── cleaned_inventory.csv    # Normalized and typed dataset
│
├── models/
│   ├── demand_model.pkl             # Trained Random Forest Regressor bundle
│   ├── stockout_model.pkl           # Trained Random Forest Classifier bundle
│   ├── product_cluster_model.pkl    # Trained Product K-Means + PCA bundle
│   ├── customer_cluster_model.pkl   # Trained Customer RFM K-Means bundle
│   └── anomaly_model.pkl            # Trained Isolation Forest bundle
│
├── src/
│   ├── data_loader.py               # Dataset generation, CSV loading, schema validation
│   ├── preprocessing.py             # Cleaning, type coercion, deduplication, EDA summary
│   ├── feature_engineering.py       # Lag features, stock ratios, RFM extraction
│   ├── demand_forecasting.py        # Time-aware split, regressor training, recursive forecasting
│   ├── stockout_prediction.py       # Classifier training, probability calibration, risk alerts
│   ├── product_clustering.py        # K-Means product segmentation, Elbow, Silhouette, PCA
│   ├── customer_segmentation.py     # RFM clustering and persona interpretation
│   ├── anomaly_detection.py         # Isolation Forest and natural-language explanations
│   ├── inventory_engine.py          # ROP, Safety Stock, EOQ, Prescriptive Action generator
│   └── model_evaluation.py          # Unified model diagnostics and metric reporting
│
├── dashboard/
│   ├── overview.py                  # Page 1: Executive Overview & KPI cards
│   ├── forecasting.py               # Page 2: Demand Forecasting & Scenario Simulator
│   ├── inventory.py                 # Page 3: Inventory Intelligence & Prescriptions
│   ├── product_clusters.py          # Page 4: Product Clustering & 2D PCA
│   ├── customer_segments.py         # Page 5: Customer RFM Segmentation
│   ├── anomalies.py                 # Page 6: Anomaly Detection Timeline & Audit
│   ├── model_performance.py         # Page 7: Model Evaluation, Confusion Matrix & Feature Importances
│   └── data_explorer.py             # Page 8: CSV Upload, Schema Validation & Correlation Heatmap
│
└── notebooks/
    └── exploratory_analysis.ipynb   # Interactive analysis and ML experimentation notebook
```

---

## 🛠️ Technology Stack

- **Core Runtime:** Python 3
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (RandomForestRegressor, RandomForestClassifier, KMeans, IsolationForest, PCA, StandardScaler, OneHotEncoder)
- **Model Serialization:** Joblib
- **Visualization:** Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
- **User Interface:** Streamlit (Custom SaaS CSS theme)

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your workstation.

### 2. Clone and Navigate to Project
```bash
cd "/Users/divyjain/Desktop/ipl predictor"
```

### 3. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline (Optional – Pretrained models are included)
To regenerate the synthetic dataset and retrain all 5 models:
```bash
PYTHONPATH=. python3 src/data_loader.py
PYTHONPATH=. python3 src/preprocessing.py
PYTHONPATH=. python3 src/demand_forecasting.py
PYTHONPATH=. python3 src/stockout_prediction.py
PYTHONPATH=. python3 src/product_clustering.py
PYTHONPATH=. python3 src/customer_segmentation.py
PYTHONPATH=. python3 src/anomaly_detection.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser to the local URL (typically `http://localhost:8501`).

---

## 📊 Dashboard Pages Walkthrough

1. **Executive Overview:**
   - Real-time KPI metric cards (Total SKUs, Warehouse Valuation, Sales Revenue, Units Sold, Avg Profit Margin, High Risk SKUs).
   - Revenue and units sold momentum over time with daily/weekly/monthly aggregation toggles.
   - Category revenue share (donut chart) and inventory distribution.
   - Top-selling products and monthly revenue vs. gross profit comparisons.
   - Overall stock health risk breakdown.

2. **Demand Forecasting:**
   - Product selection dropdown (by SKU ID and Name).
   - Historical sales trend overlaid with validation holdout predictions.
   - Interactive 7-day and 30-day recursive forecasts with simulated warehouse stock drawdown.
   - Day-of-week and monthly seasonality indices.
   - **Interactive Scenario Simulator:** Adjust price, discount, current stock, and lead time to evaluate live model predictions.

3. **Inventory Intelligence:**
   - Multi-factor inventory recommendations (*Reorder Immediately*, *Monitor Inventory*, *Overstock Detected*, *Slow-Moving Product*, *High-Demand Product*).
   - Quadrant matrix: Stock Turnover vs. Profit Margin.
   - Supplier lead-time reliability and risk exposure analysis.
   - Interactive inventory table with category, supplier, and risk filters, plus direct CSV export.

4. **Product Clustering:**
   - K-Means product grouping with 2D PCA projection coordinates.
   - Hyperparameter validation: Elbow method curve (inertia) and Silhouette score curve.
   - Cluster characteristics table with automated behavioral interpretations.
   - Individual SKU cluster assignment browser.

5. **Customer Segmentation:**
   - RFM behavioral cluster mapping (Recency, Frequency, Monetary).
   - Spend vs. Purchase Frequency scatter plot and 2D PCA customer projection.
   - Average Order Value (AOV) and discount sensitivity breakdown.
   - Prescriptive CRM retention action playbook for each customer persona.

6. **Anomaly Detection:**
   - Isolation Forest timeline comparing normal transactions with flagged anomalies.
   - Anomaly severity score histogram and category distribution.
   - Detailed inspection register with natural-language root-cause explanations (e.g. demand surges, failed discounts, severe stockouts).

7. **Model Performance:**
   - **Demand Regressor:** MAE, RMSE, R² Score on chronological test split, plus Feature Importance rankings.
   - **Stockout Classifier:** Accuracy, Precision, Recall, Macro F1, ROC-AUC, Confusion Matrix heatmap, and Feature Importances.
   - **Clustering:** Elbow curves, Silhouette coefficients, and variance explained.
   - **Isolation Forest:** Contamination rate, normal vs. anomalous proportions, and algorithmic transparency notes.

8. **Data Explorer:**
   - Upload custom enterprise CSV files with automated schema verification and column validation.
   - Filter and search records by category or product.
   - Interactive Pearson correlation heatmap for numeric features.
   - Missing value and null percentage audit table.
   - Download filtered datasets and exported reports.

---

## 🔒 Data Integrity & Best Practices

- **Zero Temporal Data Leakage:** Demand forecasting uses a strict chronological train/test split. All rolling window and lag features are shifted by at least one period so future sales are never visible during feature computation.
- **Dynamic Cluster Profiling:** Cluster personas are derived from calculated cluster-wise statistical rankings rather than hardcoded labels.
- **Synthetic vs. Real Data Transparency:** Synthetic datasets are explicitly identified with status badges and disclaimer notices, while allowing seamless integration with custom enterprise CSV uploads.
