import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import plotly.express as px

# =============================
# Config
# =============================
st.set_page_config(page_title="Mapa municipal — Mortalidad materna", layout="wide")
st.title("Mapa municipal — Mortalidad materna")

# =============================
# Helpers
# =============================
def zfill_cvegeo(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(".0", "", regex=False)
    s = s.str.replace(r"\D", "", regex=True)   # solo dígitos
    if len(s) == 0:
        return s
    s = s.str[-5:]                              # por si se alarga
    s = s.str.zfill(5)
    return s

def norm_cvegeo_value(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    s = s.replace(".0", "")
    s = re.sub(r"\D", "", s)  # solo dígitos
    if s == "":
        return None
    if len(s) > 5:
        s = s[-5:]
    return s.zfill(5)

@st.cache_data
def load_main():
    return pd.read_csv("data/municipios_mortalidad_materna.csv")

@st.cache_data
def load_geojson():
    with open("data/municipios_mexico_simplificado.geojson", "r", encoding="utf-8") as f:
        gj = json.load(f)

    # Normaliza properties.CVEGEO a str de 5 dígitos
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        # Si trae CVEGEO en otra variante, intenta rescatarlo
        if "CVEGEO" in props:
            props["CVEGEO"] = norm_cvegeo_value(props.get("CVEGEO"))
        elif "cvegeo" in props:
            props["CVEGEO"] = norm_cvegeo_value(props.get("cvegeo"))
        elif "CVE_MUN" in props and "CVE_ENT" in props:
            props["CVEGEO"] = norm_cvegeo_value(str(props.get("CVE_ENT")) + str(props.get("CVE_MUN")))
        else:
            props["CVEGEO"] = None

        feat["properties"] = props

    return gj

# =============================
# Data
# =============================
df = load_main()
geojson_mun = load_geojson()

if "CVEGEO" not in df.columns:
    st.error("No encuentro la columna 'CVEGEO' en tu CSV.")
    st.stop()

df["CVEGEO"] = zfill_cvegeo(df["CVEGEO"])

ent_col = "nom_ent" if "nom_ent" in df.columns else None
mun_col = "MUNICIPIO" if "MUNICIPIO" in df.columns else None

# Años disponibles (según columnas)
possible_years = [2020, 2021, 2022, 2023, 2024]
years_found = [y for y in possible_years if (f"mm{y}" in df.columns) or (f"rmm{y}" in df.columns)]
if not years_found:
    years_found = possible_years

# =============================
# Sidebar
# =============================
st.sidebar.header("Filtros")

anio = st.sidebar.selectbox("Año", years_found, index=len(years_found) - 1)

metric = st.sidebar.radio("Métrica", ["Muertes maternas", "RMM"], index=0)

if ent_col:
    entidades = ["Todas"] + sorted(df[ent_col].dropna().unique().tolist())
    ent_sel = st.sidebar.selectbox("Entidad", entidades, index=0)
else:
    ent_sel = "Todas"
    st.sidebar.info("No encontré 'nom_ent' en el CSV; se mostrará todo el país sin filtro por entidad.")

# Opcional: ocultar ceros para que “se vea” más el mapa
hide_zeros = st.sidebar.checkbox("Ocultar municipios con valor 0", value=False)

# =============================
# Columns per year
# =============================
col_mm = f"mm{anio}" if f"mm{anio}" in df.columns else None
col_rmm = f"rmm{anio}" if f"rmm{anio}" in df.columns else None
col_nac = f"nacimientos{anio}" if f"nacimientos{anio}" in df.columns else None

if metric == "Muertes maternas":
    if not col_mm:
        st.error(f"No existe la columna {col_mm} en tu CSV.")
        st.stop()
    map_col = col_mm
    map_title = f"Muertes maternas {anio}"
else:
    if not col_rmm:
        # Si no hay rmmYYYY, intenta construirla con mm/nacimientos
        if col_mm and col_nac:
            df[col_rmm] = (pd.to_numeric(df[col_mm], errors="coerce") / pd.to_numeric(df[col_nac], errors="coerce")) * 100000
        else:
            st.error(f"No existe la columna rmm{anio} y tampoco hay mm{anio}/nacimientos{anio} para construirla.")
            st.stop()
    map_col = col_rmm
    map_title = f"RMM {anio}"

# =============================
# Filter DF
# =============================
df_f = df.copy()
if ent_col and ent_sel != "Todas":
    df_f = df_f[df_f[ent_col] == ent_sel].copy()

df_f[map_col] = pd.to_numeric(df_f[map_col], errors="coerce").fillna(0)

# Si ocultas ceros, conviértelos a NaN (no se pintan)
plot_df = df_f.copy()
if hide_zeros:
    plot_df.loc[plot_df[map_col] == 0, map_col] = np.nan

# =============================
# Diagnostics (match)
# =============================
geo_ids = {feat["properties"].get("CVEGEO") for feat in geojson_mun.get("features", [])}
geo_ids.discard(None)
df_ids = set(plot_df["CVEGEO"].dropna().astype(str))
matched = df_ids.intersection(geo_ids)

with st.expander("Diagnóstico (match CVEGEO)", expanded=False):
    st.write("Municipios en selección (df):", len(df_ids))
    st.write("Municipios en GeoJSON:", len(geo_ids))
    st.write("Coincidencias (match):", len(matched))
    st.write("Ejemplo df CVEGEO:", sorted(list(df_ids))[:10])
    st.write("Ejemplo geojson CVEGEO:", sorted(list(geo_ids))[:10])

# =============================
# MAP
# =============================
st.subheader("Mapa")

# Rango de color: si hay muchos ceros, ayuda acotar a percentiles
vals = plot_df[map_col].dropna()
if len(vals) > 0:
    vmin = float(np.nanpercentile(vals, 5))
    vmax = float(np.nanpercentile(vals, 95))
    if vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
else:
    vmin, vmax = None, None

fig = px.choropleth(
    plot_df,
    geojson=geojson_mun,
    locations="CVEGEO",
    featureidkey="properties.CVEGEO",
    color=map_col,
    color_continuous_scale="Reds",
    hover_name=mun_col if mun_col else "CVEGEO",
    hover_data={
        "CVEGEO": True,
        (ent_col if ent_col else "CVEGEO"): True,
        map_col: True
    },
    range_color=(vmin, vmax) if (vmin is not None and vmax is not None and vmin < vmax) else None,
)

# IMPORTANTÍSIMO para que NO se vea negro y sí se vea el país:
fig.update_layout(
    height=700,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor="white",
    plot_bgcolor="white",
    coloraxis_colorbar=dict(title=map_title),
)

# Bordes + fondo gris claro (para que aunque los municipios sean “blancos”, se distingan)
fig.update_traces(marker_line_width=0.4, marker_line_color="#777777")

fig.update_geos(
    visible=True,
    showcountries=True,
    countrycolor="#999999",
    showland=True,
    landcolor="#EFEFEF",
    showocean=True,
    oceancolor="#DDEAF2",
    showlakes=True,
    lakecolor="#DDEAF2",
    bgcolor="white",
    projection_type="mercator",
    fitbounds="locations",  # con "Todas", esto ajusta a México completo (si tu geojson es México)
)

# Si estás en "Todas", NO quieras que se vaya a “cualquier lugar raro”
# (y si seleccionas estado, igual hace zoom automáticamente por fitbounds)
st.plotly_chart(fig, use_container_width=True)

