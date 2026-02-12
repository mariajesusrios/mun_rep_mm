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

def cap_outliers(series: pd.Series, upper_q: float = 0.99) -> pd.Series:
    """Recorta valores extremos (winsorización) para que la escala del mapa sea usable."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    upper = s.quantile(upper_q)
    return s.clip(upper=upper)

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

# USAREMOS SOLO EL SIMPLIFICADO (más estable para Streamlit)
geojson_simpl = load_geojson("data/municipios_mexico_simplificado.geojson")
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

years_found = [y for y in [2020, 2021, 2022, 2023, 2024]
               if (f"mm{y}" in df.columns or f"rmm{y}" in df.columns)]
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
# OJO: ocultar_ceros NO debe borrar polígonos si queremos ver fronteras.
# Aquí lo usaremos solo para "no colorear", pero manteniendo el borde.

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
            st.error("No existe RMM y no puedo calcularla (falta mm o nacimientos).")
            st.stop()
    else:
        map_val_col = col_rmm
    map_title = f"RMM {anio}"

df_work[map_val_col] = pd.to_numeric(df_work[map_val_col], errors="coerce")

# 🔥 CRÍTICO: cap outliers en RMM para que el mapa nacional sea legible
if metric == "RMM":
    df_work[map_val_col] = cap_outliers(df_work[map_val_col], upper_q=0.99)

# =========================
# DATAFRAME PARA PLOT (ESQUELETO + LEFT JOIN)
# =========================
plot_df = geo_base.copy()

if ent_code_sel is not None:
    plot_df = plot_df[plot_df["CVE_ENT"] == ent_code_sel].copy()

cols_to_merge = ["CVEGEO", map_val_col]
if ent_col: cols_to_merge.append(ent_col)
if mun_col: cols_to_merge.append(mun_col)

plot_df = plot_df.merge(df_work[cols_to_merge], on="CVEGEO", how="left")

# Valor para pintar: NaN -> 0 para que el polígono exista.
plot_df["_plot_value_"] = plot_df[map_val_col].fillna(0)

# Si quieren "ocultar ceros": no quitamos polígonos, solo los hacemos muy claros
# logramos eso haciendo el color_range arrancar en 0 y usando un borde visible.
# (Si conviertes 0 a NaN, Plotly NO dibuja ese polígono y pierdes fronteras.)
# =========================
# MAPA
# =========================
st.subheader(f"Mapa municipal — {anio}")

geojson_use = geojson_simpl

hover_data = {map_val_col: True, "CVEGEO": True}
if ent_col:
    hover_data[ent_col] = True

fig = px.choropleth(
    plot_df,
    geojson=geojson_use,
    locations="CVEGEO",
    featureidkey="properties.CVEGEO",
    color="_plot_value_",
    color_continuous_scale="Reds",
    hover_name=mun_col if mun_col else "CVEGEO",
    hover_data=hover_data,
)

# Bordes SIEMPRE visibles
fig.update_traces(marker_line_color="rgba(60,60,60,0.70)", marker_line_width=0.35)

# ✅ ZOOM/EXTENT ROBUSTO:
# - Estado: fitbounds a locations
# - Todas: fijar bbox México (evita "se ve negro / se pierde el país")
if ent_code_sel is not None:
    fig.update_geos(fitbounds="locations")
else:
    # bbox aprox de México (WGS84)
    fig.update_geos(
        center=dict(lat=23.6, lon=-102.5),
        projection_scale=5.2,
        lataxis_range=[14.0, 33.5],
        lonaxis_range=[-118.5, -86.0],
    )

# Fondo claro (no negro)
fig.update_geos(
    visible=False,
    showland=True, landcolor="rgb(245,245,245)",
    showocean=True, oceancolor="rgb(230,230,230)",
)

# Si "ocultar ceros": hacemos que el rango de color empiece en 0
# y controlamos el rango superior con cuantiles para que se note variación.
# (Mantiene fronteras)
vals = plot_df["_plot_value_"].astype(float)
if vals.notna().sum() > 0:
    if metric == "RMM":
        upper = np.nanquantile(vals.values, 0.99)
    else:
        upper = np.nanmax(vals.values)
    if np.isfinite(upper) and upper > 0:
        fig.update_layout(coloraxis=dict(cmin=0, cmax=float(upper)))

fig.update_layout(
    height=720,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title=map_title),
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DIAGNÓSTICO
# =========================
with st.expander("Diagnóstico (match CVEGEO)", expanded=False):
    geo_ids = {feat.get("properties", {}).get("CVEGEO") for feat in geojson_use.get("features", [])}
    geo_ids.discard(None)
    df_ids = set(plot_df["CVEGEO"].dropna().astype(str))
    matched = df_ids.intersection(geo_ids)

    st.write("Municipios en selección (plot_df):", len(df_ids))
    st.write("Municipios en GeoJSON:", len(geo_ids))
    st.write("Coincidencias (match):", len(matched))
    st.write("Ejemplo plot_df CVEGEO:", sorted(list(df_ids))[:10])
    st.write("Ejemplo geojson CVEGEO:", sorted(list(geo_ids))[:10])
