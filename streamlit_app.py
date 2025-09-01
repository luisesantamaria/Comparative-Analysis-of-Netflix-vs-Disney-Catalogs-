# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Netflix vs Disney+ – Catalog Explorer", layout="wide")

@st.cache_data
def load_data():
    # Rutas relativas dentro del repo
    nf_path = "data/netflix_titles.csv"
    dz_path = "data/disney_plus_titles.csv"

    # Carga robusta
    try:
        df_nf = pd.read_csv(nf_path)
    except Exception as e:
        st.error(f"Could not read {nf_path}: {e}")
        raise
    try:
        df_dz = pd.read_csv(dz_path)
    except Exception as e:
        st.error(f"Could not read {dz_path}: {e}")
        raise

    # Etiqueta de plataforma (por si no existe)
    if "platform" not in df_nf.columns:
        df_nf["platform"] = "Netflix"
    if "platform" not in df_dz.columns:
        df_dz["platform"] = "Disney+"

    # Unifica columnas clave si hay variantes de nombre
    # (en tus CSV ya vienen como en el notebook, pero dejamos mapeo defensivo)
    rename_map = {
        "listed_in": "genres",
        "description": "description",
        "release_year": "release_year",
        "title": "title",
        "type": "type",
        "duration": "duration",
        "country": "country",
        "rating": "rating",
    }
    df_nf = df_nf.rename(columns=rename_map)
    df_dz = df_dz.rename(columns=rename_map)

    df = pd.concat([df_nf, df_dz], ignore_index=True)

    # Normaliza 'type'
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.title()
        df["type"] = df["type"].replace({"Tv Show": "TV Show", "Tv Show": "TV Show"})

    # Feature engineering simple de duración/seasons según tu proyecto
    def extract_minutes(x):
        if isinstance(x, str) and "min" in x:
            return pd.to_numeric(x.split()[0], errors="coerce")
        return np.nan

    def extract_seasons(x):
        if isinstance(x, str) and "Season" in x:
            return pd.to_numeric(x.split()[0], errors="coerce")
        return np.nan

    if "duration" in df.columns:
        df["duration_min"] = df["duration"].apply(extract_minutes)
        df["seasons_n"] = df["duration"].apply(extract_seasons)

    # Limpieza básica de año
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df = df[df["release_year"].between(1900, 2035)]

    # Primary genre
    if "genres" in df.columns:
        def split_clean(s):
            if pd.isna(s): return []
            return [t.strip() for t in str(s).split(",") if t.strip()]
        df["genres_list"] = df["genres"].apply(split_clean)
        df["primary_genre"] = df["genres_list"].apply(lambda xs: xs[0] if xs else np.nan)

    return df

df = load_data()

# ===== Sidebar filters =====
st.sidebar.header("Filters")
platforms = st.sidebar.multiselect(
    "Platform", sorted(df["platform"].dropna().unique().tolist()),
    default=sorted(df["platform"].dropna().unique().tolist())
)
types = st.sidebar.multiselect(
    "Type", sorted(df["type"].dropna().unique().tolist()),
    default=sorted(df["type"].dropna().unique().tolist())
)
year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())
year_range = st.sidebar.slider("Release year", min_value=year_min, max_value=year_max,
                               value=(max(year_min, 2000), year_max))

mask = (
    df["platform"].isin(platforms) &
    df["type"].isin(types) &
    df["release_year"].between(year_range[0], year_range[1])
)
df_f = df.loc[mask].copy()

# ===== Header =====
st.title("Netflix vs Disney+ — Catalog Explorer")
st.caption("Quick, reproducible view of the comparative EDA results. Data loaded from `/data` in this repo.")

# ===== KPIs =====
c1, c2, c3, c4 = st.columns(4)
c1.metric("Titles", f"{len(df_f):,}")
c2.metric("Movies share", f"{(df_f['type'].eq('Movie').mean()*100):.1f}%")
c3.metric("Median movie runtime", f"{df_f.loc[df_f['type'].eq('Movie'), 'duration_min'].median():.0f} min")
c4.metric("Median TV seasons", f"{df_f.loc[df_f['type'].eq('TV Show'), 'seasons_n'].median():.1f}")

# ===== Charts =====
tab1, tab2, tab3 = st.tabs(["By platform & type", "Release trend", "Top genres"])

with tab1:
    st.subheader("Counts by platform and type")
    st.bar_chart(
        df_f.groupby(["platform", "type"]).size().rename("count")
    )

with tab2:
    st.subheader("Titles released by year")
    st.line_chart(
        df_f.groupby(["release_year", "platform"]).size().rename("count").unstack(fill_value=0)
    )

with tab3:
    st.subheader("Top-15 primary genres (current filter)")
    topg = (df_f["primary_genre"]
            .value_counts().head(15)
            .rename_axis("primary_genre").reset_index(name="count"))
    st.dataframe(topg, use_container_width=True)

# ===== Raw sample =====
st.markdown("### Sample (first 20 rows)")
st.dataframe(df_f.head(20), use_container_width=True)
