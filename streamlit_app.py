import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import re

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Mapa municipal — Mortalidad materna", layout="wide")
st.title("Mapa municipal — Mortalidad materna")

# ============================================================
# HELPERS
# ============================================================
def zfill_cvegeo(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(".0", "", regex=False)
    s = s.str.replace(r"\D", "", regex=True)   # solo dígitos
    s = s.str[-5:]                             # por si se alarga
    s = s.str.zfill(5)
    return s

def coerce_int_flag(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.replace({"True": 1, "False": 0, True: 1, False: 0})
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return s

@st.cache_data
def load_main():
    # Ajusta si tu CSV está en otra ruta
    return pd.read_csv("data/municipios_mortalidad_materna.csv")

@st.cache_data
def load_geojson():
    # ESTE debe ser el geojson “INEGI” que trae CVEGEO municipal
    with open("data/municipios_mexico.geojson", "r", encoding="utf-8") as f:
        gj = json.load(f)

    def norm_cvegeo(val):
        s = str(val).strip()
        s = s.replace(".0", "")
        s = re.sub(r"\D", "", s)   # solo dígitos
        if len(s) > 5:
            s = s[-5:]
        return s.zfill(5)

    for feat in gj.get("features", []):
        props = feat.get("properties", {})

        if "CVEGEO" in props and props["CVEGEO"] is not None:
            props["CVEGEO"] = norm_cvegeo(props["CVEGEO"])
        elif "cvegeo" in props and props["cvegeo"] is not None:
            props["CVEGEO"] = norm_cvegeo(props["cvegeo"])
        elif "CVE_MUN" in props and "CVE_ENT" in props:
            props["CVEGEO"] = norm_cvegeo(str(props["CVE_ENT"]) + str(props["CVE_MUN"]))
        else:
            props["CVEGEO"] = None

        feat["properties"] = props

    return gj

# ============================================================
# LOAD
# ============================================================
df = load_main()
geojson_mun = load_geojson()

if "CVEGEO" not in df.columns:
    st.error("No encuentro la columna 'CVEGEO' en tu CSV.")
    st.stop()

df["CVEGEO"] = zfill_cvegeo(df["CVEGEO"])

# columnas opcionales
ent_col = "nom_ent" if "nom_ent" in df.columns else None
mun_col = "MUNICIPIO" if "MUNICIPIO" in df.columns else None

# años disponibles por columnas
years_found = []
for y in [2020, 2021, 2022, 2023, 2024]:
    if f"rmm{y}" in df.columns or f"mm{y}" in df.columns:
        years_found.append(y)
if not years_found:
    years_found = [2020, 2021, 2022, 2023, 2024]

# criterios para clasificar 1/2/3
crit_cols = [c for c in ["criterio1", "criterio2", "criterio3"] if c in df.columns]
has_criteria = len(crit_cols) == 3
if has_criteria:
    for c in crit_cols:
        df[c] = coerce_int_flag(df[c])
    df["n_criterios"] = df[crit_cols].sum(axis=1).astype(int)
else:
    df["n_criterios"] = np.nan

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Filtros")

anio = st.sidebar.selectbox("Año", years_found)

col_rmm = f"rmm{anio}" if f"rmm{anio}" in df.columns else None
col_mm  = f"mm{anio}" if f"mm{anio}" in df.columns else None
col_nac = f"nacimientos{anio}" if f"nacimientos{anio}" in df.columns else None

if col_rmm is None and col_mm is None:
    st.sidebar.error(f"No encuentro columnas rmm{anio} ni mm{anio} en tu CSV.")
    st.stop()

metric = st.sidebar.radio("Métrica", ["Muertes maternas", "RMM"], index=0)

# entidad
if ent_col:
    entidades = ["Todas"] + sorted(df[ent_col].dropna().unique().tolist())
    ent_sel = st.sidebar.selectbox("Entidad", entidades)
else:
    ent_sel = "Todas"

# filtro repetidores por n_criterios
if has_criteria:
    filtro_rep = st.sidebar.selectbox(
        "Municipios repetidores",
        ["Todos", "Cumplen 1 criterio", "Cumplen 2 criterios", "Cumplen 3 criterios"]
    )
else:
    filtro_rep = "Todos"
    st.sidebar.warning("No encontré criterio1/2/3; no se puede filtrar por 1/2/3 criterios.")

ocultar_ceros = st.sidebar.checkbox("Ocultar municipios con valor 0", value=False)

# ============================================================
# FILTERS
# ============================================================
df_f = df.copy()

if ent_col and ent_sel != "Todas":
    df_f = df_f[df_f[ent_col] == ent_sel]

if has_criteria:
    if filtro_rep == "Cumplen 1 criterio":
        df_f = df_f[df_f["n_criterios"] == 1]
    elif filtro_rep == "Cumplen 2 criterios":
        df_f = df_f[df_f["n_criterios"] == 2]
    elif filtro_rep == "Cumplen 3 criterios":
        df_f = df_f[df_f["n_criterios"] == 3]

# ============================================================
# KPIs ARRIBA
# ============================================================
mun_total = int(df_f.shape[0])

mm_total = np.nan
if col_mm:
    mm_total = pd.to_numeric(df_f[col_mm], errors="coerce").sum()

nac_total = np.nan
if col_nac:
    nac_total = pd.to_numeric(df_f[col_nac], errors="coerce").sum()

# RMM agregada preferida: (Σ muertes / Σ nacimientos)*100000
rmm_agregada = np.nan
if col_mm and col_nac and pd.notna(nac_total) and nac_total > 0:
    rmm_agregada = (mm_total / nac_total) * 100000
elif col_rmm:
    rmm_agregada = pd.to_numeric(df_f[col_rmm], errors="coerce").mean()

k1, k2, k3 = st.columns(3)
k1.metric("Municipios (selección)", f"{mun_total:,}")
k2.metric(f"Muertes maternas {anio}", f"{int(mm_total):,}" if pd.notna(mm_total) else "-")
k3.metric(f"RMM agregada {anio}", "-" if np.isnan(rmm_agregada) else f"{rmm_agregada:,.1f}")

if col_mm and col_nac:
    st.caption("RMM agregada = (Σ muertes / Σ nacimientos) × 100,000.")

st.divider()

# ============================================================
# MAPA
# ============================================================
st.subheader(f"Mapa municipal — {anio}")

# define color column
if metric == "Muertes maternas":
    if col_mm is None:
        st.error(f"No existe la columna mm{anio} en tu CSV.")
        st.stop()
    map_color_col = col_mm
    map_title = f"Muertes maternas {anio}"
else:
    # RMM: si no existe rmmYYYY pero hay mm/nac, la calculamos
    if col_rmm is None:
        if col_mm and col_nac:
            col_rmm = f"rmm_calc_{anio}"
            df_f[col_rmm] = (pd.to_numeric(df_f[col_mm], errors="coerce") /
                             pd.to_numeric(df_f[col_nac], errors="coerce")) * 100000
        else:
            st.error(f"No existe rmm{anio} y tampoco puedo calcularla (falta mm{anio} o nacimientos{anio}).")
            st.stop()
    map_color_col = col_rmm
    map_title = f"RMM {anio}"

map_df = df_f.copy()
map_df[map_color_col] = pd.to_numeric(map_df[map_color_col], errors="coerce")

# opcional: ocultar ceros
if ocultar_ceros:
    map_df = map_df[map_df[map_color_col].fillna(0) != 0]

n_valid = map_df[map_color_col].notna().sum()
if n_valid == 0:
    st.warning(f"No hay valores numéricos en {map_color_col} para la selección actual.")
else:
    hover_data = {map_color_col: True, "CVEGEO": True}
    if ent_col:
        hover_data[ent_col] = True
    if has_criteria:
        hover_data["n_criterios"] = True

    fig_map = px.choropleth(
        map_df,
        geojson=geojson_mun,
        locations="CVEGEO",
        featureidkey="properties.CVEGEO",
        color=map_color_col,
        color_continuous_scale="Reds",
        hover_name=mun_col if mun_col else "CVEGEO",
        hover_data=hover_data,
    )

    # ✅ CLAVE: si es "Todas", NO usar fitbounds="locations"
    if ent_col and ent_sel != "Todas":
        fig_map.update_geos(fitbounds="locations")
    else:
        fig_map.update_geos(
            scope="north america",
            center={"lat": 23.5, "lon": -102.0},
            projection_scale=4.7
        )

    fig_map.update_geos(visible=False, projection_type="mercator")
    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=680,
        coloraxis_colorbar=dict(title=map_title),
    )

    st.plotly_chart(fig_map, use_container_width=True, key=f"map_{anio}_{metric}_{ent_sel}_{filtro_rep}_{ocultar_ceros}")

# ============================================================
# DIAGNÓSTICO CVEGEO
# ============================================================
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
