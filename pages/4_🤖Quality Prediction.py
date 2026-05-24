import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from style import (
    apply_futuristic_style, page_header, section_header, chart_label,
    kpi_card, sidebar_brand, insight, footer, plotly_futuristic_layout,
    COLOR_SEQ, COLORS, df_table,
)

st.set_page_config(
    page_title="Quality Prediction — Wine Quality",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_futuristic_style()
sidebar_brand()

page_header("🤖 Wine Quality Prediction", subtitle="Random Forest ML model for real-time quality scoring")

FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide",
            "density", "pH", "sulphates", "alcohol"]


@st.cache_resource
def train_model():
    df = pd.read_csv("Wine Quality Dataset.csv")
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    X = df[FEATURES]
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    importances = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, acc, importances, df


model, accuracy, importances, df = train_model()

col1, col2, col3, col4 = st.columns(4)
kpi_card(col1, "Model Accuracy",    f"{accuracy*100:.1f}%",   color="#10B981")
kpi_card(col2, "Training Samples",  f"{int(len(df)*0.8):,}",  color="#06B6D4")
kpi_card(col3, "Test Samples",      f"{int(len(df)*0.2):,}",  color="#8B5CF6")
kpi_card(col4, "Total Features",    f"{len(FEATURES)}",       color="#F59E0B")

st.markdown("---")

tab1, tab2 = st.tabs(["🎛️ Predict Quality", "📊 Model Insights"])

with tab1:
    section_header("🌲 Enter Wine Chemical Properties")
    chart_label(
        "Adjust the sliders below to set each chemical value, then click Predict.",
        "All values are bounded by the real data range from the Wine Quality Dataset.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.slider(
            "Fixed Acidity",
            float(df["fixed acidity"].min()), float(df["fixed acidity"].max()), 7.4, step=0.1,
        )
        volatile_acidity = st.slider(
            "Volatile Acidity",
            float(df["volatile acidity"].min()), float(df["volatile acidity"].max()), 0.52, step=0.01,
        )
        citric_acid = st.slider(
            "Citric Acid",
            float(df["citric acid"].min()), float(df["citric acid"].max()), 0.26, step=0.01,
        )
        residual_sugar = st.slider(
            "Residual Sugar",
            float(df["residual sugar"].min()), float(df["residual sugar"].max()), 2.2, step=0.1,
        )

    with col2:
        chlorides = st.slider(
            "Chlorides",
            float(df["chlorides"].min()), float(df["chlorides"].max()), 0.079, step=0.001,
            format="%.3f",
        )
        free_sulfur = st.slider(
            "Free Sulfur Dioxide",
            float(df["free sulfur dioxide"].min()), float(df["free sulfur dioxide"].max()), 15.0, step=1.0,
        )
        total_sulfur = st.slider(
            "Total Sulfur Dioxide",
            float(df["total sulfur dioxide"].min()), float(df["total sulfur dioxide"].max()), 46.0, step=1.0,
        )

    with col3:
        density = st.slider(
            "Density",
            float(df["density"].min()), float(df["density"].max()), 0.9968, step=0.0001,
            format="%.4f",
        )
        ph = st.slider(
            "pH",
            float(df["pH"].min()), float(df["pH"].max()), 3.31, step=0.01,
        )
        sulphates = st.slider(
            "Sulphates",
            float(df["sulphates"].min()), float(df["sulphates"].max()), 0.58, step=0.01,
        )
        alcohol = st.slider(
            "Alcohol",
            float(df["alcohol"].min()), float(df["alcohol"].max()), 10.2, step=0.1,
        )

    if st.button("🍷 Predict Wine Quality", use_container_width=True):
        input_data = pd.DataFrame(
            [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
              chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]],
            columns=FEATURES,
        )

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = np.max(proba) * 100

        if prediction >= 7:
            quality_status = "Excellent Quality 🍷"
            border_color   = "#00F5FF"
            glow_color     = "rgba(0,245,255,0.4)"
        elif prediction >= 5:
            quality_status = "Good Quality ✅"
            border_color   = "#22C55E"
            glow_color     = "rgba(34,197,94,0.4)"
        else:
            quality_status = "Low Quality ⚠️"
            border_color   = "#EF4444"
            glow_color     = "rgba(239,68,68,0.4)"

        result_html = f"""
        <div style="
            background: linear-gradient(135deg,#0F172A,#1E293B);
            padding: 32px;
            border-radius: 22px;
            text-align: center;
            border: 2px solid {border_color};
            box-shadow: 0 0 28px {glow_color};
            margin: 20px auto;
            max-width: 560px;
        ">
            <h2 style="color:{border_color};margin-bottom:10px;font-size:20px;font-weight:700;
                       font-family:'Rajdhani',sans-serif;letter-spacing:0.06em;text-transform:uppercase;">
                🍷 Predicted Wine Quality
            </h2>
            <h1 style="color:white;font-size:72px;margin:0;font-weight:900;
                       font-family:'Orbitron',sans-serif;line-height:1;color:{border_color};">
                {prediction}
            </h1>
            <h3 style="color:#e2e8f0;margin-top:12px;font-size:20px;font-weight:600;
                       font-family:'Rajdhani',sans-serif;">
                {quality_status}
            </h3>
            <p style="color:#CBD5E1;font-size:15px;margin-top:14px;">
                Model Confidence: <b style="color:{border_color};">{confidence:.1f}%</b>
            </p>
            <p style="color:#64748b;font-size:12px;margin-top:8px;font-family:'Rajdhani',sans-serif;">
                Random Forest · 200 Estimators · {len(FEATURES)} Features
            </p>
        </div>
        """
        st.html(result_html)

        classes = model.classes_
        proba_df = pd.DataFrame({"Quality": classes, "Probability (%)": (proba * 100).round(1)})
        fig_proba = px.bar(
            proba_df, x="Quality", y="Probability (%)",
            color="Probability (%)", color_continuous_scale="Turbo",
            text="Probability (%)",
        )
        fig_proba.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_color="#cbd5e1")
        fig_proba.update_layout(**plotly_futuristic_layout("Prediction Probability per Quality Class"))
        fig_proba.update_xaxes(title="Quality Score")
        fig_proba.update_yaxes(title="Probability (%)", range=[0, 100])
        st.plotly_chart(fig_proba, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        section_header("🏆 Feature Importance")
        chart_label("Top Predictors", "Which chemical properties drive the model's quality predictions most")
        fig_imp = px.bar(
            importances, x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale="Turbo",
            text=importances["Importance"].map(lambda v: f"{v:.3f}"),
        )
        fig_imp.update_traces(textposition="outside", textfont_color="#cbd5e1")
        fig_imp.update_layout(**plotly_futuristic_layout(height=420))
        fig_imp.update_xaxes(title="Importance Score")
        fig_imp.update_yaxes(title="")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        section_header("📊 Feature Importance Table")
        chart_label("Ranked Features", "All features ranked by their contribution to quality predictions")
        imp_display = importances.copy()
        imp_display["Importance"] = imp_display["Importance"].map(lambda v: f"{v:.4f}")
        imp_display["Rank"] = range(1, len(imp_display) + 1)
        imp_display = imp_display[["Rank", "Feature", "Importance"]]
        df_table(imp_display, show_index=False)

    section_header("📈 Actual vs Predicted Quality Distribution")
    chart_label("Model Evaluation", "Compares the real quality distribution with model predictions on test data")

    X_all = df[FEATURES]
    y_all = df["quality"]
    _, X_test_all, _, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    y_pred_all = model.predict(X_test_all)

    actual_counts = pd.Series(y_test_all).value_counts().sort_index().reset_index()
    actual_counts.columns = ["Quality", "Count"]
    actual_counts["Type"] = "Actual"

    pred_counts = pd.Series(y_pred_all).value_counts().sort_index().reset_index()
    pred_counts.columns = ["Quality", "Count"]
    pred_counts["Type"] = "Predicted"

    compare_df = pd.concat([actual_counts, pred_counts])
    fig_comp = px.bar(
        compare_df, x="Quality", y="Count",
        color="Type", barmode="group",
        color_discrete_map={"Actual": "#00F5FF", "Predicted": "#8B5CF6"},
        text="Count",
    )
    fig_comp.update_traces(textposition="outside", textfont_color="#cbd5e1")
    fig_comp.update_layout(**plotly_futuristic_layout(height=400))
    fig_comp.update_xaxes(title="Quality Score")
    fig_comp.update_yaxes(title="Count")
    st.plotly_chart(fig_comp, use_container_width=True)

    insight(
        f"Model achieves <b>{accuracy*100:.1f}% accuracy</b> on the held-out test set (20% of data).",
        label="Model Performance", kind="positive",
    )
    insight(
        "Alcohol is the most important predictor — higher alcohol content strongly drives quality scores up.",
        label="Top Feature", kind="positive",
    )
    insight(
        "Volatile acidity is the second most important feature — even small increases hurt quality predictions.",
        label="Key Negative Signal", kind="warning",
    )
    insight(
        "Imbalanced class distribution (few 3 and 8-rated wines) may slightly reduce accuracy at extremes.",
        label="Class Imbalance Note", kind="info",
    )

footer()
