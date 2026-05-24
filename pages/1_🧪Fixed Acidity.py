
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
from data import load_data, apply_global_filters
import plotly.express as px

st.set_page_config(page_title="Wine Quality",initial_sidebar_state="expanded")
apply_futuristic_style()
sidebar_brand()


df = load_data("Wine Quality Dataset.csv")
df = apply_global_filters(df)

mean_value=df["fixed acidity"].mean()
median_value=df["fixed acidity"].median()
std_value=df["fixed acidity"].std()
min_value=df["fixed acidity"].min()
max_value=df["fixed acidity"].max()

page_header("🧪 Fixed Acidity")

col1,col2,col3,col4,col5 = st.columns(5)
kpi_card(col1,"Mean",f"{mean_value:,.2f}%",color="#06B6D4")
kpi_card(col2,"Median",f"{median_value:,.2f}%",color="#EC4899")
kpi_card(col3,"Standard Deviation",f"{std_value:,.2f}",color="#F97316")
kpi_card(col4,"Minimum Values",f"{min_value:,}",color="#8B5CF6")
kpi_card(col5,"Maximum Values",f"{max_value:,}",color="#3D10A8")

st.markdown("---")
tab1,tab2,tab3=st.tabs(["📊 Distribution Analysis","📈 Relationship Insights","🧠 Statistical & Quality Insights"])
with tab1:
    col1,col2=st.columns(2)
    with col1:
        section_header("📊 Fixed Acidity Distribution")
        chart_label("Histogram Analysis","Shows how fixed acidity values are distributed across wine samples")
        fig1 = px.histogram(df,x="fixed acidity",nbins=20,color_discrete_sequence=["#F59E0B"])
        fig1.update_traces(marker_line_color="#FFE4C4",marker_line_width=1.5,opacity=0.75)
        fig1.update_layout(**plotly_futuristic_layout())
        fig1.update_xaxes(title="Fixed Acidity")
        fig1.update_yaxes(title="Count")
        st.plotly_chart(fig1,width="stretch")

    with col2:
        section_header("📊 Fixed Acidity Central Tendency")
        chart_label("Mean & Median Analysis","Highlights average (mean) and central value (median) of fixed acidity")
        fig2 = px.histogram(df,x="fixed acidity",nbins=20,color_discrete_sequence=["#F59E0B"])
        fig2.update_traces(marker_line_color="#FFE4C4",marker_line_width=1.5,opacity=0.75)
        fig2.add_vline(x=mean_value,line_width=3,line_dash="dash",line_color="#00F5FF",annotation_text=f"Mean: {mean_value:.2f}",annotation_position="top right")
        fig2.add_vline(x=mean_value,line_width=3,line_dash="dot",line_color="#EC4899",annotation_text=f"Median: {median_value:.2f}",annotation_position="top left")
        fig2.update_layout(**plotly_futuristic_layout())
        fig2.update_xaxes(title="Fixed Acidity")
        fig2.update_yaxes(title="Count")
        st.plotly_chart(fig2,width="stretch")

with tab2:
        section_header("📈 Fixed Acidity vs Quality")
        chart_label("Relationship Analysis","Shows how fixed acidity impacts wine quality scores")
        fig3=px.scatter(df,x="fixed acidity",y="quality",color="quality",size="alcohol")
        fig3.update_layout(**plotly_futuristic_layout())
        fig3.update_xaxes(title="Fixed acidity")
        fig3.update_yaxes(title="Quality")
        st.plotly_chart(fig3,width="stretch")     

        section_header("📦 Fixed Acidity Distribution by Quality")
        chart_label("Box Plot Analysis","Compares fixed acidity variation across different wine quality levels")
        fig4=px.box(df,y="fixed acidity",color="quality")
        fig4.update_layout(**plotly_futuristic_layout())
        fig4.update_yaxes(title="Fixed Acidity")
        st.plotly_chart(fig4,width="stretch")   

with tab3:
    section_header("🔗 Fixed Acidity vs Alcohol")
    chart_label("Correlation Insight","Analyzes relationship between fixed acidity and alcohol content")
    fig5=px.scatter(df,x="fixed acidity",y="quality")
    fig5.update_layout(**plotly_futuristic_layout())
    st.plotly_chart(fig5,use_container_width=True)
    
    col1,col2=st.columns(2)
    with col1:
        section_header("🍷 Wine Quality Distribution")
        chart_label("Target Variable Analysis","Distribution of wine quality ratings across dataset")
        quality_counts=df["quality"].value_counts().reset_index()
        quality_counts.columns = ["quality","count"]
        fig6=px.pie(quality_counts,names="quality",values="count",hole=0.4)
        fig6.update_layout(**plotly_futuristic_layout())
        st.plotly_chart(fig6,width="stretch")

    with col2:
        section_header("📊 Quality Breakdown Table")
        df_table(quality_counts.sort_values(by="quality",ascending=False),show_index=False)
    
insight(f"Lowest value of the fixed acidity in the alcohol : {min_value:.2f}",label="Lowest Level",kind="warning")
insight(f"Highest level of the fixed acidity in the alcohol : {max_value:.2f}",label="Highest Level",kind="positive")
insight(f"<b>High fixed acidity</b> directly affect the wine quality",label="⚠ Insight")

footer()