# 🍷 Wine Quality Dashboard

An interactive Streamlit dashboard for exploring and predicting wine quality using chemical properties.

## 📁 Project Structure

```
wine_quality_dashboard/
├── Home.py                              # Main dashboard & overview
├── data.py                              # Data loading & global filters
├── style.py                             # Futuristic dark theme & UI components
├── requirements.txt                     # Python dependencies
├── Wine Quality Dataset.csv             # ← Place your dataset here
└── pages/
    ├── 1_🧪_Fixed_Acidity.py           # Fixed acidity analysis
    ├── 2_☢️_Acidity_Intelligence_Hub.py # Multi-feature acidity hub
    ├── 3_⚠️_Sulfur_Chemical_Analysis.py # SO₂, density, chlorides
    └── 4_🤖_Quality_Prediction.py       # Random Forest ML prediction
```

## 🚀 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your dataset**  
   Place `Wine Quality Dataset.csv` in the root folder (same level as `Home.py`).

3. **Run the app**
   ```bash
   streamlit run Home.py
   ```

## 📊 Pages

| Page | Description |
|------|-------------|
| 🍷 Dashboard | Dataset overview, KPIs, correlation heatmap |
| 🧪 Fixed Acidity | Distribution, central tendency, quality impact |
| ☢️ Acidity Hub | Multi-feature acidity analysis & scatter matrix |
| ⚠️ Sulfur & Chemicals | SO₂, density, chloride, sulphates analysis |
| 🤖 Quality Prediction | Real-time quality prediction with feature importance |

## 🛠️ Technologies

- **Streamlit** — Web app framework
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — Random Forest model

## 📋 Dataset Columns

`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`,
`free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality`
