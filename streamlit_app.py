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

def norm_cvegeo(val):
    s = str(val).strip().replace(".0", "")
    s = re.sub(r"\D", "", s)
    if len(s) > 5:
        s = s[-5:]
    return s.zfill(5)

@st.cache_data
def load_main():
    return pd.read_csv("data/municipios_mortalidad_materna.csv")

@st.cache_data
def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # normaliza CVEGEO dentro del geojson
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

# ✅ Carga ambos geojson:
geojson_full = load_geojson("data/municipios_mexico.geojson")
geojson_simpl = load_geojson("data/municipios_mexico_simplificado.geojson")  # <-- asegúrate de subirlo

# ✅ base para join: usa el SIMPLIFICADO (mismo CVEGEO, menos pesado)
geo_base = geo_cvegeo_df(geojson_simpl)

if "CVEGEO" not in df.columns:
    st.error("No encuentro la columna 'CVEGEO' en tu CSV.")
    st.stop()

df["CVEGEO"] = zfill_cvegeo(df["CVEGEO"])

ent_col = "nom_ent" if "nom_ent" in df.columns else None
mun_col = "MUNICIPIO" if "MUNICIPIO" in df.columns else None
cod_ent_col = "cod_ent_s" if "cod_ent_s" in df.columns else None
if cod_ent_col:
    df[cod_ent_col] = zfill_2(df[cod_ent_col])

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

ent_sel = "Todas"
ent_code_sel = None
if ent_col and cod_ent_col:
    ent_map = df[[ent_col, cod_ent_col]].dropna().drop_duplicates().sort_values(ent_col)
    entidades = ["Todas"] + ent_map[ent_col].tolist()
    ent_sel = st.sidebar.selectbox("Entidad", entidades)
    if ent_sel != "Todas":
        ent_code_sel = ent_map.loc[ent_map[ent_col] == ent_sel, cod_ent_col].iloc[0]

ocultar_ceros = st.sidebar.checkbox("Ocultar municipios con valor 0", value=False)

# =========================
# MÉTRICA
# =========================
df_work = df.copy()
if metric == "Muertes maternas":
    if col_mm is None:
        st.error(f"No existe mm{anio}.")
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
            st.error("No existe RMM y no puedo calcularla.")
            st.stop()
    else:
        map_val_col = col_rmm
    map_title = f"RMM {anio}"

df_work[map_val_col] = pd.to_numeric(df_work[map_val_col], errors="coerce")

# =========================
# DATAFRAME PARA PLOT (ESQUELETO + LEFT JOIN)
# =========================
plot_df = geo_base.copy()

# si seleccionan estado, filtra el esqueleto para que se vean TODOS sus municipios
if ent_code_sel is not None:
    plot_df = plot_df[plot_df["CVE_ENT"] == ent_code_sel].copy()

cols_to_merge = ["CVEGEO", map_val_col]
if ent_col: cols_to_merge.append(ent_col)
if mun_col: cols_to_merge.append(mun_col)

plot_df = plot_df.merge(df_work[cols_to_merge], on="CVEGEO", how="left")

# valor para pintar (NaN => 0 para que se dibujen fronteras)
plot_df["_plot_value_"] = plot_df[map_val_col].fillna(0)

# OJO: NO elimines filas si quieres fronteras
if ocultar_ceros:
    # no quitamos filas; solo dejamos 0
    pass

# =========================
# MAPA
# =========================
st.subheader(f"Mapa municipal — {anio}")

# ✅ GeoJSON que se usa:
# - "Todas": simplificado (para que renderice)
# - Estado: también puede ser simplificado (recomendado). Si quieres, cambia a geojson_full.
geojson_use = geojson_simpl if ent_code_sel is None else geojson_simpl

fig = px.choropleth(
    plot_df,
    geojson=geojson_use,
    locations="CVEGEO",
    featureidkey="properties.CVEGEO",
    color="_plot_value_",
    color_continuous_scale="Reds",
    hover_name=mun_col if mun_col else "CVEGEO",
    hover_data={map_val_col: True, "CVEGEO": True, ent_col: True} if ent_col else {map_val_col: True, "CVEGEO": True},
)

# fronteras visibles
fig.update_traces(marker_line_color="rgba(80,80,80,0.75)", marker_line_width=0.35)

# fitbounds correcto
if ent_code_sel is not None:
    fig.update_geos(fitbounds="locations")
else:
    fig.update_geos(fitbounds="geojson")

# fondo claro
fig.update_geos(
    visible=False,
    showland=True, landcolor="rgb(245,245,245)",
    showocean=True, oceancolor="rgb(230,230,230)",
)

fig.update_layout(
    height=720,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title=map_title),
)

st.plotly_chart(fig, use_container_width=True)
