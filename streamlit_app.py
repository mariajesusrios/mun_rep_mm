import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import re

st.set_page_config(page_title="Mapa municipal — Mortalidad materna", layout="wide")
st.title("Mapa municipal — Mortalidad materna")

# =========================
# HELPERS
# =========================
def zfill_cvegeo(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(".0", "", regex=False)
    s = s.str.replace(r"\D", "", regex=True)
    s = s.str[-5:]
    return s.str.zfill(5)

@st.cache_data
def load_main():
    return pd.read_csv("data/municipios_mortalidad_materna.csv")

@st.cache_data
def load_geojson_4326():
    """
    IMPORTANTE:
    Usa el archivo YA reproyectado a EPSG:4326.
    """
    with open("data/municipios_mexico_4326.geojson", "r", encoding="utf-8") as f:
        gj = json.load(f)

    def norm_cvegeo(val):
        s = str(val).strip().replace(".0", "")
        s = re.sub(r"\D", "", s)
        if len(s) > 5:
            s = s[-5:]
        return s.zfill(5)

    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if "CVEGEO" in props and props["CVEGEO"] is not None:
            props["CVEGEO"] = norm_cvegeo(props["CVEGEO"])
        elif "cvegeo" in props and props["cvegeo"] is not None:
            props["CVEGEO"] = norm_cvegeo(props["cvegeo"])
        else:
            props["CVEGEO"] = None
        feat["properties"] = props

    return gj

# =========================
# LOAD DATA
# =========================
df = load_main()
geojson_mun = load_geojson_4326()

if "CVEGEO" not in df.columns:
    st.error("No encuentro la columna 'CVEGEO' en tu CSV.")
    st.stop()

df["CVEGEO"] = zfill_cvegeo(df["CVEGEO"])

ent_col = "nom_ent" if "nom_ent" in df.columns else None
mun_col = "MUNICIPIO" if "MUNICIPIO" in df.columns else None

# años disponibles
years_found = []
for y in [2020, 2021, 2022, 2023, 2024]:
    if f"mm{y}" in df.columns or f"rmm{y}" in df.columns:
        years_found.append(y)
if not years_found:
    years_found = [2020, 2021, 2022, 2023, 2024]

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Filtros")
anio = st.sidebar.selectbox("Año", years_found)

col_mm = f"mm{anio}" if f"mm{anio}" in df.columns else None
col_rmm = f"rmm{anio}" if f"rmm{anio}" in df.columns else None
col_nac = f"nacimientos{anio}" if f"nacimientos{anio}" in df.columns else None

metric = st.sidebar.radio("Métrica", ["Muertes maternas", "RMM"], index=0)

if ent_col:
    entidades = ["Todas"] + sorted(df[ent_col].dropna().unique().tolist())
    ent_sel = st.sidebar.selectbox("Entidad", entidades)
else:
    ent_sel = "Todas"

ocultar_ceros = st.sidebar.checkbox("Ocultar municipios con valor 0", value=False)

# =========================
# FILTER
# =========================
df_f = df.copy()
if ent_col and ent_sel != "Todas":
    df_f = df_f[df_f[ent_col] == ent_sel]

# =========================
# METRIC COLUMN
# =========================
if metric == "Muertes maternas":
    if col_mm is None:
        st.error(f"No existe la columna mm{anio} en tu CSV.")
        st.stop()
    map_color_col = col_mm
    map_title = f"Muertes maternas {anio}"
else:
    # Si no existe rmmYYYY pero sí mm/nac, la calculamos
    if col_rmm is None:
        if col_mm and col_nac:
            col_rmm = f"rmm_calc_{anio}"
            df_f[col_rmm] = (
                pd.to_numeric(df_f[col_mm], errors="coerce") /
                pd.to_numeric(df_f[col_nac], errors="coerce")
            ) * 100000
        else:
            st.error(f"No existe rmm{anio} y no puedo calcularla (falta mm{anio} o nacimientos{anio}).")
            st.stop()
    map_color_col = col_rmm
    map_title = f"RMM {anio}"

map_df = df_f.copy()
map_df[map_color_col] = pd.to_numeric(map_df[map_color_col], errors="coerce")

if ocultar_ceros:
    map_df = map_df[map_df[map_color_col].fillna(0) != 0]

# =========================
# MAP
# =========================
st.subheader(f"Mapa municipal — {anio}")

n_valid = map_df[map_color_col].notna().sum()
if n_valid == 0:
    st.warning(f"No hay valores numéricos en {map_color_col} para la selección actual.")
    st.stop()

hover_data = {map_color_col: True, "CVEGEO": True}
if ent_col:
    hover_data[ent_col] = True

fig = px.choropleth(
    map_df,
    geojson=geojson_mun,
    locations="CVEGEO",
    featureidkey="properties.CVEGEO",
    color=map_color_col,
    color_continuous_scale="Reds",
    hover_name=mun_col if mun_col else "CVEGEO",
    hover_data=hover_data,
)

# ✅ CLAVE PARA QUE SE VEA EL PAÍS COMPLETO:
# Fit a TODO el geojson (no a "locations" filtradas) cuando es "Todas"
if ent_col and ent_sel != "Todas":
    fig.update_geos(fitbounds="locations")
else:
    fig.update_geos(fitbounds="geojson")

# Muestra “contexto” (tierra y costa) para que NO se vea “negro”
fig.update_geos(
    visible=False,
    showcountries=True,
    countrycolor="rgba(120,120,120,0.6)",
    showland=True,
    landcolor="rgb(245,245,245)",
    showocean=True,
    oceancolor="rgb(230,230,230)",
)

fig.update_layout(
    height=720,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title=map_title),
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DIAGNÓSTICO
# =========================
geo_ids = {feat.get("properties", {}).get("CVEGEO") for feat in geojson_mun.get("features", [])}
geo_ids.discard(None)
df_ids = set(df_f["CVEGEO"].dropna().astype(str))
matched = df_ids.intersection(geo_ids)

with st.expander("Diagnóstico (match CVEGEO)", expanded=False):
    st.write("Municipios en selección (df):", len(df_ids))
    st.write("Municipios en GeoJSON:", len(geo_ids))
    st.write("Coincidencias (match):", len(matched))
    st.write("Ejemplo df CVEGEO:", sorted(list(df_ids))[:10])
    st.write("Ejemplo geojson CVEGEO:", sorted(list(geo_ids))[:10])
