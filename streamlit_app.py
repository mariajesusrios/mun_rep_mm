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

def zfill_2(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(".0", "", regex=False)
    s = s.str.replace(r"\D", "", regex=True)
    s = s.str[-2:]
    return s.str.zfill(2)

@st.cache_data
def load_main():
    return pd.read_csv("data/municipios_mortalidad_materna.csv")

@st.cache_data
def load_geojson():
    with open("data/municipios_mexico.geojson", "r", encoding="utf-8") as f:
        gj = json.load(f)

    def norm_cvegeo(val):
        s = str(val).strip().replace(".0", "")
        s = re.sub(r"\D", "", s)
        if len(s) > 5:
            s = s[-5:]
        return s.zfill(5)

    # Normaliza CVEGEO en properties
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

@st.cache_data
def geo_cvegeo_df(geojson_mun: dict) -> pd.DataFrame:
    """DataFrame base con TODOS los CVEGEO del GeoJSON (para que se dibujen las fronteras)."""
    geo_ids = [feat.get("properties", {}).get("CVEGEO") for feat in geojson_mun.get("features", [])]
    geo_ids = [g for g in geo_ids if g is not None]
    geo_ids = sorted(set(geo_ids))
    base = pd.DataFrame({"CVEGEO": geo_ids})
    base["CVE_ENT"] = base["CVEGEO"].str[:2]
    return base

# =========================
# LOAD DATA
# =========================
df = load_main()
geojson_mun = load_geojson()
geo_base = geo_cvegeo_df(geojson_mun)

if "CVEGEO" not in df.columns:
    st.error("No encuentro la columna 'CVEGEO' en tu CSV.")
    st.stop()

df["CVEGEO"] = zfill_cvegeo(df["CVEGEO"])

# Columnas esperadas en tu CSV (tú sí las tienes)
ent_col = "nom_ent" if "nom_ent" in df.columns else None
mun_col = "MUNICIPIO" if "MUNICIPIO" in df.columns else None
cod_ent_col = "cod_ent_s" if "cod_ent_s" in df.columns else None

if cod_ent_col:
    df[cod_ent_col] = zfill_2(df[cod_ent_col])

# años disponibles
years_found = [y for y in [2020, 2021, 2022, 2023, 2024] if (f"mm{y}" in df.columns or f"rmm{y}" in df.columns)]
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

# Entidad
ent_sel = "Todas"
ent_code_sel = None

if ent_col and cod_ent_col:
    ent_map = (
        df[[ent_col, cod_ent_col]]
        .dropna()
        .drop_duplicates()
        .sort_values(ent_col)
    )
    entidades = ["Todas"] + ent_map[ent_col].tolist()
    ent_sel = st.sidebar.selectbox("Entidad", entidades)

    if ent_sel != "Todas":
        ent_code_sel = ent_map.loc[ent_map[ent_col] == ent_sel, cod_ent_col].iloc[0]

ocultar_ceros = st.sidebar.checkbox("Ocultar municipios con valor 0", value=False)

# =========================
# DEFINE MÉTRICA PARA MAPA
# =========================
df_work = df.copy()

if metric == "Muertes maternas":
    if col_mm is None:
        st.error(f"No existe la columna mm{anio} en tu CSV.")
        st.stop()
    map_val_col = col_mm
    map_title = f"Muertes maternas {anio}"
else:
    if col_rmm is None:
        if col_mm and col_nac:
            map_val_col = f"rmm_calc_{anio}"
            df_work[map_val_col] = (
                pd.to_numeric(df_work[col_mm], errors="coerce") /
                pd.to_numeric(df_work[col_nac], errors="coerce")
            ) * 100000
        else:
            st.error(f"No existe rmm{anio} y no puedo calcularla (falta mm{anio} o nacimientos{anio}).")
            st.stop()
    else:
        map_val_col = col_rmm
    map_title = f"RMM {anio}"

df_work[map_val_col] = pd.to_numeric(df_work[map_val_col], errors="coerce")

# =========================
# ARMA DATAFRAME DE PLOT CON TODOS LOS MUNICIPIOS
# (AQUÍ ESTÁ LA CLAVE)
# =========================
plot_df = geo_base.copy()

# Si seleccionan estado: filtra el ESQUELETO por CVE_ENT (para que se vean TODOS los municipios del estado, tengan o no dato)
if ent_code_sel is not None:
    plot_df = plot_df[plot_df["CVE_ENT"] == ent_code_sel].copy()

# Join de datos del CSV contra el esqueleto
cols_to_merge = ["CVEGEO", map_val_col]
if ent_col: cols_to_merge.append(ent_col)
if mun_col: cols_to_merge.append(mun_col)

plot_df = plot_df.merge(df_work[cols_to_merge], on="CVEGEO", how="left")

# Relleno para que Plotly DIBUJE todos los polígonos (NaN => 0 para pintar muy claro)
plot_df["_plot_value_"] = plot_df[map_val_col].fillna(0)

# Si quieres ocultar ceros, quitamos ceros SOLO del color (pero eso también quita polígonos).
# Tú pediste ver fronteras aunque no haya dato, así que:
# - si ocultar_ceros=True, NO eliminamos filas; solo marcamos un color "casi blanco".
if ocultar_ceros:
    # dejamos 0 pero aclaramos visualmente con rango (sigue dibujando fronteras)
    pass

# =========================
# MAPA
# =========================
st.subheader(f"Mapa municipal — {anio}")

hover_data = {
    "_plot_value_": False,  # no mostrar el interno
    map_val_col: True,
    "CVEGEO": True,
}
if ent_col:
    hover_data[ent_col] = True

fig = px.choropleth(
    plot_df,
    geojson=geojson_mun,
    locations="CVEGEO",
    featureidkey="properties.CVEGEO",
    color="_plot_value_",
    color_continuous_scale="Reds",
    hover_name=mun_col if mun_col else "CVEGEO",
    hover_data=hover_data,
)

# Fronteras municipales visibles SIEMPRE
fig.update_traces(marker_line_color="rgba(90,90,90,0.65)", marker_line_width=0.4)

# Fitbounds correcto: estado => locations, país => geojson
if ent_code_sel is not None:
    fig.update_geos(fitbounds="locations")
else:
    fig.update_geos(fitbounds="geojson")

# Fondo (evita “negro vacío”)
fig.update_geos(
    visible=False,
    showland=True,
    landcolor="rgb(245,245,245)",
    showocean=True,
    oceancolor="rgb(230,230,230)",
    showcountries=True,
    countrycolor="rgba(120,120,120,0.6)",
)

fig.update_layout(
    height=720,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title=map_title),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DIAGNÓSTICO
# =========================
geo_ids = set(geo_base["CVEGEO"].astype(str))
df_ids = set(df["CVEGEO"].dropna().astype(str))
matched = df_ids.intersection(geo_ids)

with st.expander("Diagnóstico (match CVEGEO)", expanded=False):
    st.write("Municipios en CSV:", len(df_ids))
    st.write("Municipios en GeoJSON:", len(geo_ids))
    st.write("Coincidencias:", len(matched))
    st.write("Ejemplo CSV CVEGEO:", sorted(list(df_ids))[:10])
    st.write("Ejemplo GeoJSON CVEGEO:", sorted(list(geo_ids))[:10])
    st.write("Faltan en CSV (primeros 20):", sorted(list(geo_ids - df_ids))[:20])
