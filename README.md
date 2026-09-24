# Used Car Price Prediction

An end-to-end machine learning project that predicts used car prices from real market data.

## 📦 Dataset

`data/used_cars.csv` — 4,009 used car listings with 12 raw features including brand, model year, mileage, fuel type, engine specs, transmission, accident history, and price.

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | pandas, NumPy |
| Feature engineering | scikit-learn `ColumnTransformer` |
| Modelling | Ridge, Random Forest, Gradient Boosting, XGBoost |
| Evaluation | MAE, RMSE, R² |
| Explainability | SHAP (`TreeExplainer`) |
| Serialisation | joblib |
| Dashboard | Streamlit |
| Charts | Plotly |

## 📁 Project Structure

```
used-car-price-prediction/
├── data/
│   └── used_cars.csv
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model  (creates models/ artifacts)
python src/train.py

# 3. Launch the dashboard
streamlit run app.py
```

## 🧹 Feature Engineering

- `price` — parsed from `"$10,300"` → `10300.0`; log-transformed as target
- `mileage` — parsed from `"51,000 mi."` → `51000.0`
- `horsepower`, `engine_litres`, `cylinders` — extracted from raw engine string via regex
- `transmission` — bucketed: Automatic / Manual / CVT / Dual-Clutch / Other
- `fuel_type` — simplified: Gasoline / Hybrid / Electric / Diesel / Flex Fuel / Other
- `has_accident` — binary flag
- `car_age` — `2024 − model_year`
- Rare colours collapsed to "Other"

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Dataset KPIs + sample data |
| 📊 Market Analytics | Interactive Plotly charts (brand, year, fuel, correlation) |
| 🔮 Price Prediction | Form input → instant price estimate + gauge + similar listings |
| 🧠 Model Insights | Metrics table, feature importance, SHAP, Actual vs Predicted |
| ℹ️ About | Architecture overview |
