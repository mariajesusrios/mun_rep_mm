import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Mapa municipal — Mortalidad materna")

@st.cache_data
def load_data():
    df = pd.read_csv("data/municipios_mortalidad_materna.csv")
    df["CVEGEO"] = df["CVEGEO"].astype(str).str.zfill(5)
    return df

@st.cache_data
def load_geojson():
    with open("data/municipios_mexico_simplificado.geojson", "r", encoding="utf-8") as f:
        return json.load(f)

df = load_data()
geojson_mun = load_geojson()

anio = 2020
col_mm = f"mm{anio}"

df[col_mm] = pd.to_numeric(df[col_mm], errors="coerce")

fig = px.choropleth(
    df,
    geojson=geojson_mun,
    locations="CVEGEO",
    featureidkey="properties.CVEGEO",
    color=col_mm,
    color_continuous_scale="Reds",
)

fig.update_geos(
    visible=True,
    showcountries=False,
    showcoastlines=False,
    showland=True,
    landcolor="#F5F5F5",
    center={"lat": 23.5, "lon": -102},
    projection_scale=4.2
)

fig.update_layout(margin=dict(r=0, t=0, l=0, b=0))

st.plotly_chart(fig, use_container_width=True)
