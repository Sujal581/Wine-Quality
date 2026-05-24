
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

avg_sulfur=df["free sulfur dioxide"].mean()
max_sulfur=df["free sulfur dioxide"].max()
avg_total_sulfur=df["total sulfur dioxide"].mean()
max_total_sulfur=df["total sulfur dioxide"].max()
min_density=df["density"].min()
max_density=df["density"].max()
min_chlorides=df["chlorides"].min()
max_chlorides=df["chlorides"].max()

page_header("⚠️ Sulfur & Other Chemical Analysis")

col1,col2,col3,col4=st.columns(4)
kpi_card(col1,"Average Free SO₂",f"{avg_sulfur:.2f}%",color="#06B6D4")
kpi_card(col2,"Maximum Free SO₂",f"{max_sulfur:.2f}",color="#EC4899")
kpi_card(col3,"Average Total SO₂",f"{avg_total_sulfur:.2f}%",color="#F97316")
kpi_card(col4,"Maximum Total SO₂",f"{max_total_sulfur:.2f}",color="#8B5CF6")

col5,col6,col7,col8=st.columns(4)
kpi_card(col5,"Minimum Denisty",f"{min_density:.2f}",color="#06B6D4")
kpi_card(col6,"Maximum Denisty",f"{max_density:.2f}",color="#F59E0B")
kpi_card(col7,"Minimum Chlorides",f"{min_chlorides:.2f}",color="#10B981")
kpi_card(col8,"Maximum Chlorides",f"{max_chlorides:.2f}",color="#EF4444")

st.markdown("---")
col1,col2=st.columns(2)
with col1:
        section_header("🧪 Free SO₂ Distribution")
        chart_label("Free SO₂ Distribution","Free SO₂ helps to preserve wine")
        fig1 = px.histogram(df,x="free sulfur dioxide",nbins=10,color_discrete_sequence=["#636EFA"])
        fig1.update_layout(**plotly_futuristic_layout())
        fig1.update_xaxes(title="Free SO₂")
        st.plotly_chart(fig1,width="stretch")

with col2:
        section_header("📦 Total SO₂ by Wine Quality")
        chart_label("Total SO₂","Total SO₂ level across wine quality")
        fig2 = px.box(df,y="total sulfur dioxide",color="quality")
        fig2.update_layout(**plotly_futuristic_layout())
        fig2.update_yaxes(title="Total SO₂")
        st.plotly_chart(fig2,width="stretch")

col3,col4=st.columns(2)
with col3:
        section_header("🔬 Free SO₂ VS Total SO₂ Relationship")
        chart_label("Free SO₂ VS Total SO₂","Relationship between free SO₂ Vs total SO₂")
        fig3=px.scatter(df,x="free sulfur dioxide",y="total sulfur dioxide",color="quality")
        fig3.update_layout(**plotly_futuristic_layout())
        fig3.update_xaxes(title="Free SO₂")
        fig3.update_yaxes(title="Total SO₂")
        st.plotly_chart(fig3,width="content")     

with col4:
    section_header("⚖️ Density VS Alcohol Analysis")
    chart_label("Density VS Alcohol","Relationship between density and alcohol")
    fig4 = px.line(df.sort_values("density"), x="density", y="alcohol")
    fig4.update_layout(**plotly_futuristic_layout())
    st.plotly_chart(fig4,width="content")  
    

section_header("🧂 Chlorides VS Wine Quality Distribution")
chart_label("Chlorides VS Quality","Distribution of chlorides level across wine quality")
fig=px.violin(df,x="quality",y="chlorides",color="quality",box=True,points="all")
fig.update_layout(**plotly_futuristic_layout())
st.plotly_chart(fig,use_container_width=True)  


insight(label="SO₂ Distribution",text="Free SO₂ values are mostly concentrated in lower-to-mid ranges, indicating balanced preservation levels",kind="info")
insight(label="Quality Observation",text="Higher quality wines generally maintain controlled total SO₂ levels without extreme outliers",kind="success")
insight(label="Correlation Insight",text="Free SO₂ and Total SO₂ show a positive relationship, suggesting sulfur compounds increase together",kind="warning")
insight(label="Alcohol vs Density",text="Density tends to decrease as alcohol content increases due to fermentation effects",kind="info")
insight(label="Chlorides Analysis",text="Excessive salt content is less common in high-quality wines",kind="success")
insight(label="Distribution Pattern",text="Most wines cluster around moderate chloride concentrations in the violin distribution",kind="warning")


footer()
