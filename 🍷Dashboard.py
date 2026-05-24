import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from style import (
    apply_futuristic_style, page_header, section_header, chart_label,
    kpi_card, sidebar_brand, insight, footer, plotly_futuristic_layout,
    COLOR_SEQ, COLORS, df_table
)
import plotly.express as px
from data import load_data

st.set_page_config(page_title="Wine Quality",initial_sidebar_state="expanded")
apply_futuristic_style()
sidebar_brand()


df = load_data("Wine Quality Dataset.csv")

total_rows,total_cols = df.shape
missing_values=df.isna().sum().sum()
duplicate_rows=df.duplicated().sum()


page_header("🍷 Wine Quality Dashboard")

col1,col2,col3,col4 = st.columns(4)
kpi_card(col1,"Rows",f"{total_rows:,}",color="#00F5FF")
kpi_card(col2,"Columns",f"{total_cols:,}",color="#3B82F6")
kpi_card(col3,"Missing Values",f"{missing_values:,}",color="#EF4444")
kpi_card(col4,"Duplicate Values",f"{duplicate_rows:,}",color="#F59E0B")



st.markdown("---")
section_header("About Project:")
chart_label("""
This project analyzes different chemical properties of wine to understand their impact on wine quality and build a machine learning model for quality prediction.

The dataset contains features like fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, sulfur dioxide, density, pH, sulphates, and alcohol. Using data analysis and visualization techniques, the project identifies patterns, correlations, and important factors affecting wine quality.

The project includes:

• Data Cleaning & Preprocessing  
• Exploratory Data Analysis (EDA)  
• Interactive Visualizations  
• KPI Metrics & Insights  
• Machine Learning Models for Quality Prediction  
• Streamlit Dashboard Development  

Key Insights:
• Higher alcohol content often leads to better wine quality.  
• High volatile acidity negatively affects quality.  
• Sulphates and citric acid can positively influence wine quality.  

Technologies Used:
• Python  
• Pandas & NumPy  
• Matplotlib, Seaborn, Plotly  
• Scikit-learn  
• Streamlit  

The main objective is to create an interactive and intelligent wine quality analysis dashboard that provides insights and predicts wine quality accurately.
""")
footer()