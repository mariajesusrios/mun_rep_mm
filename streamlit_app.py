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
# MAP (TODOS LOS MUNICIPIOS + FRONTERAS)
# =========================
st.subheader(f"Mapa municipal — {anio}")

# --- 1) lista completa de municipios del GeoJSON (CVEGEO)
geo_ids = [feat.get("properties", {}).get("CVEGEO") for feat in geojson_mun.get("features", [])]
geo_ids = [g for g in geo_ids if g is not None]

base_geo = pd.DataFrame({"CVEGEO": pd.Series(geo_ids, dtype=str).dropna().unique()})
base_geo["CVEGEO"] = zfill_cvegeo(base_geo["CVEGEO"])

# --- 2) filtra tus datos por entidad (si aplica) y deja numérico
data_df = df_f[["CVEGEO", map_color_col]].copy()
data_df["CVEGEO"] = zfill_cvegeo(data_df["CVEGEO"])
data_df[map_color_col] = pd.to_numeric(data_df[map_color_col], errors="coerce")

if ocultar_ceros:
    data_df = data_df[data_df[map_color_col].fillna(0) != 0]

# --- 3) merge: conserva TODOS los municipios (aunque no tengan dato)
plot_df = base_geo.merge(data_df, on="CVEGEO", how="left")

# Si estás filtrando por estado, reduce a ese estado también (pero conservando sus municipios sin dato)
if ent_col and ent_sel != "Todas":
    # OJO: para esto necesitamos saber qué municipios del geojson pertenecen a la entidad.
    # Si tu geojson trae NOM_ENT / CVE_ENT en properties, úsalo. Si no, lo hacemos por df.
    # Aquí lo hacemos por df (más seguro con tu dataset):
    cves_estado = set(df_f["CVEGEO"].astype(str))
    plot_df = plot_df[plot_df["CVEGEO"].isin(cves_estado)]

# --- 4) Columna para pintar: NaN -> -1 (sentinela para "Sin dato")
plot_col = f"{map_color_col}__plot"
plot_df[plot_col] = plot_df[map_color_col].copy()
plot_df[plot_col] = plot_df[plot_col].where(plot_df[plot_col].notna(), -1)

# rango para color (evita que -1 “rompa” la escala)
max_val = pd.to_numeric(plot_df[map_color_col], errors="coerce").max()
if pd.isna(max_val):
    max_val = 1

# Escala: gris para -1, rojo para datos
# (0 corresponde a -1 porque ponemos range_color=(-1, max_val))
custom_scale = [
    (0.0,  "rgb(230,230,230)"),  # -1 => sin dato (gris)
    (0.000001, "rgb(255,245,240)"),
    (0.2,  "rgb(254,224,210)"),
    (0.4,  "rgb(252,146,114)"),
    (0.6,  "rgb(251,106,74)"),
    (0.8,  "rgb(222,45,38)"),
    (1.0,  "rgb(165,15,21)"),
]

fig = px.choropleth(
    plot_df,
    geojson=geojson_mun,
    locations="CVEGEO",
    featureidkey="properties.CVEGEO",
    color=plot_col,
    color_continuous_scale=custom_scale,
    range_color=(-1, max_val),
    hover_name=mun_col if mun_col else "CVEGEO",
    hover_data={
        "CVEGEO": True,
        map_color_col: True,
        plot_col: False,   # no mostrar la columna sentinela
    },
)

# --- 5) fronteras visibles SIEMPRE
fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(120,120,120,0.7)")

# --- 6) encuadre país/estado
fig.update_geos(projection_type="mercator")

if ent_col and ent_sel != "Todas":
    fig.update_geos(fitbounds="locations")
else:
    fig.update_geos(center=dict(lat=23.5, lon=-102.0), projection_scale=4.8)

# --- 7) fondo / contexto
fig.update_geos(
    showland=True, landcolor="rgb(245,245,245)",
    showocean=True, oceancolor="rgb(230,230,230)",
    showcoastlines=True, coastlinecolor="rgba(150,150,150,0.7)",
)

fig.update_layout(
    height=720,
    margin=dict(r=0, t=0, l=0, b=0),
    coloraxis_colorbar=dict(title=map_title),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)
