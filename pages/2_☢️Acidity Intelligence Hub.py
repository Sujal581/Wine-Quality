
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from style import (
    apply_futuristic_style, page_header, section_header, chart_label,
    kpi_card, sidebar_brand, insight, footer, plotly_futuristic_layout,
    COLOR_SEQ, COLORS, df_table
)
import plotly.express as px
from data import load_data, apply_global_filters

st.set_page_config(page_title="Wine Quality",initial_sidebar_state="expanded")
apply_futuristic_style()
sidebar_brand()


df = load_data("Wine Quality Dataset.csv")
df = apply_global_filters(df)

acidity = ["fixed acidity","volatile acidity","citric acid"]
mean_value=df[acidity].mean().mean()
median_value=df[acidity].median().median()
std_value=df[acidity].std().mean().mean()
min_value=df[acidity].min().min()
max_value=df[acidity].max().max()

page_header("☢️ Acidity Analysis")

col1,col2,col3,col4,col5 = st.columns(5)
kpi_card(col1,"Mean",f"{mean_value:.2f}%",color="#06B6D4")
kpi_card(col2,"Median",f"{median_value:.2f}%",color="#EC4899")
kpi_card(col3,"Standard Deviation",f"{std_value:.2f}",color="#F97316")
kpi_card(col4,"Minimum Values",f"{min_value:.2f}",color="#8B5CF6")
kpi_card(col5,"Maximum Values",f"{max_value:.2f}",color="#8B5CF6")

st.markdown("---")
tab1, tab2,tab3 = st.tabs(["📊 Feature Distribution", "📈 Relationships & Correlation","🧪 Feature Deep Dive"])
with tab1:
        section_header("📊 Fixed Acidity Distribution")
        chart_label("Distribution Overview","Shows how fixed acidity values are distributed across wine samples")
        fig1 = px.scatter(df,x="fixed acidity",y="quality",color="quality",size="quality")
        fig1.update_traces(marker_line_color="#FFE4C4",marker_line_width=1.5,opacity=0.75)
        fig1.update_layout(**plotly_futuristic_layout())
        fig1.update_xaxes(title="Fixed Acidity")
        fig1.update_yaxes(title="Quality")
        st.plotly_chart(fig1,width="stretch")

        section_header("📊 Volatile Acidity Distribution")
        chart_label("Feature Variation","Shows how volatile acidity changes across wine quality levels")
        fig2 = px.scatter(df,x="volatile acidity",y="quality",color="quality",size="quality")
        fig2.update_traces(marker_line_color="#FFE4C4",marker_line_width=1.5,opacity=0.75)
        fig2.update_layout(**plotly_futuristic_layout())
        fig2.update_xaxes(title="Volatile Acidity")
        fig2.update_yaxes(title="Quality")
        st.plotly_chart(fig2,width="stretch")

        insight("📊 Distribution Pattern: Fixed acidity shows moderate spread but clusters around mid-range values",label="Fixed Acidity",kind="warning")
        insight("🍇 Quality Signal: Volatile acidity is mostly concentrated in lower ranges, which is good for wine quality",label="Volatile Acidity",kind="positive")
        insight("🎯 Quality Trend: Higher quality wines tend to avoid extreme acidity values",label="Quality Pattern",kind="positive")

with tab2:
        col1,col2=st.columns(2)
        with col1:
            section_header("📈 Citric Acid VS Quality")
            chart_label("Relationship Analysis","Shows how citric acid varies with wine quality")
            fig3=px.box(df,y="citric acid",color="quality")
            fig3.update_layout(**plotly_futuristic_layout())
            fig3.update_yaxes(title="Citric Acid")
            st.plotly_chart(fig3,width="content")     

        with col2:
            section_header("📦 Acidity Correlation Heatmap")
            chart_label("Correlation Analysis","Shows relationships between acidity features")
            acidity_df=df[["fixed acidity","citric acid","volatile acidity","pH","quality"]]
            corr_matrix=acidity_df.corr()
            fig4=px.imshow(corr_matrix,color_continuous_scale="Turbo")
            fig4.update_layout(**plotly_futuristic_layout())
            st.plotly_chart(fig4,width="content")   
    
        section_header("🧪 pH Variation Across Wine Quality")
        chart_label("Distribution Insight","Shows how pH levels vary across different wine quality ratings")
        fig = px.violin(df, x="quality", y="pH", box=True, color="quality")
        fig.update_layout(**plotly_futuristic_layout())
        st.plotly_chart(fig, use_container_width=True)

        insight("🧪 Citric Insight: Citric acid shows weak-to-moderate positive relationship with quality", label="Citric Acid Impact",kind="positive")
        insight("⚖️ Stability Check: pH levels remain relatively stable across quality classes", label="pH Stability",kind="positive")
        insight("🔗 Correlation Alert: Strong correlations exist between acidity-related features (multicollinearity present)", label="Feature Correlation",kind="warning")
    
with tab3:
        section_header("🍬 Residual Sugar Impact on Quality")
        chart_label("Relationship Insight","Analyzes how residual sugar influences wine quality scores")
        fig = px.scatter(df,x="residual sugar",y="quality",color="quality",size="alcohol")
        fig.update_layout(**plotly_futuristic_layout())
        fig.update_xaxes(title="Residual Sugar")
        fig.update_yaxes(title="Quality")
        st.plotly_chart(fig, use_container_width=True)

        section_header("📊 Multi-Feature Relationship Matrix")
        chart_label("Feature Interaction Analysis","Shows how key chemical properties interact with each other")
        fig = px.scatter_matrix(df,dimensions=["citric acid", "alcohol", "pH"],color="quality")
        fig.update_layout(**plotly_futuristic_layout())
        st.plotly_chart(fig, use_container_width=True)
        
        section_header("📦 Acidity Outlier Detection")
        chart_label("Statistical Insight","Identifies variation and outliers in acidity-related features across wine quality levels")
        fig = px.box(df,y=["fixed acidity", "volatile acidity", "citric acid"],color="quality")
        fig.update_layout(**plotly_futuristic_layout(""))
        fig.update_xaxes(title="Variables")
        fig.update_yaxes(title="Values")
        st.plotly_chart(fig, use_container_width=True)

        insight("🍬 Weak Influence: Residual sugar has weak influence on wine quality compared to alcohol content", label="Sugar Impact",kind="positive")
        insight("🍷 Strong Signal: Alcohol shows stronger separation between high and low quality wines", label="Alcohol Effect",kind="positive")
        insight("⚠️ Outliers Detected: Data contains noticeable outliers in acidity features", label="Data Quality",kind="warning")
        insight("🤖 ML Friendly: Feature matrix shows overlapping relationships → suitable for Random Forest / XGBoost", label="Model Insight",kind="positive")

footer()
