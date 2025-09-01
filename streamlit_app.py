import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==============
# CONFIG
# ==============
st.set_page_config(page_title="Netflix vs Disney+ — Catalog Dashboard", layout="wide")

# ==============
# HELPERS (caching + small FE utils)
# ==============
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file, encoding="utf-8", low_memory=False, on_bad_lines="skip")

def parse_duration_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Create duration_min for movies and seasons_n for TV shows with your confirmed patterns."""
    df = df.copy()
    if "type" not in df.columns or "duration" not in df.columns:
        return df

    t = df["type"].astype(str).str.strip().str.lower()
    dur = df["duration"].astype(str)

    # Movies: ... 'min' in lowercase
    is_movie = t.eq("movie") & dur.str.contains(r"\bmin\b", case=True, na=False)
    mins = pd.to_numeric(dur.str.extract(r"(\d+)\s*min", expand=False), errors="coerce")
    df["duration_min"] = np.where(is_movie, mins, np.nan)

    # TV Shows: 'SEASON'/'SEASONS' uppercase
    is_tv = t.str.contains("tv", na=False)
    seas = pd.to_numeric(dur.str.extract(r"(\d+)\s*SEASON", flags=re.IGNORECASE, expand=False), errors="coerce")
    df["seasons_n"] = np.where(is_tv, seas, np.nan)

    return df

def build_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "genres" not in df.columns:
        return df
    g = df["genres"].astype(str).replace({"": np.nan, "nan": np.nan})
    # Split by comma
    df["genres_list"] = g.dropna().apply(lambda s: [x.strip() for x in s.split(",") if x.strip()]) if g.notna().any() else np.nan
    # Primary genre = first in list
    def first_or_nan(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return np.nan
    df["primary_genre"] = df.get("genres_list").apply(first_or_nan) if "genres_list" in df.columns else np.nan
    # Genre count
    def count_or_nan(lst):
        return len(lst) if isinstance(lst, list) else np.nan
    df["genre_count"] = df.get("genres_list").apply(count_or_nan) if "genres_list" in df.columns else np.nan
    return df

def build_country_primary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "country" not in df.columns:
        return df
    c = df["country"].astype(str).replace({"": np.nan, "nan": np.nan})
    def first_country(s):
        if pd.isna(s):
            return np.nan
        parts = [x.strip() for x in str(s).split(",") if x.strip()]
        return parts[0] if parts else np.nan
    df["country_primary"] = c.apply(first_country)
    return df

def normalize_rating(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rating" not in df.columns:
        df["rating_norm"] = "Unknown"
        return df
    r = df["rating"].astype(str).str.upper().str.strip()

    def map_rating(x):
        # Mapeo simple y robusto (ajusta según tu dataset)
        if x in {"G", "TV-G", "TVY", "TV-Y", "TV-Y7", "Y7"}:
            return "G"
        if x in {"PG", "TV-PG", "TV-14"} or "PG" in x:
            return "Teen"  # agrupa PG/PG-13/TV-14 en teen
        if x in {"R", "TV-MA", "NC-17", "MA"}:
            return "Mature"
        if x in {"UNRATED", "NOT RATED", "NR", ""}:
            return "Unknown"
        # Labels raros (ej. TBG, TBGPS) -> mapea a PG/Teen por conservador
        if "TBG" in x:
            return "Teen"
        return "Other"

    df["rating_norm"] = r.apply(map_rating)
    return df

def build_decade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "release_year" not in df.columns:
        return df
    y = pd.to_numeric(df["release_year"], errors="coerce")
    df["decade"] = (np.floor(y / 10) * 10).astype("Int64")
    return df

def ensure_platform_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "platform" in df.columns:
        df["platform"] = df["platform"].astype(str).str.strip()
    if "type" in df.columns:
        # Normaliza a 'Movie' / 'TV Show'
        t = df["type"].astype(str).str.strip().str.lower()
        df["type"] = np.where(t.str.contains("movie"), "Movie",
                       np.where(t.str.contains("tv"), "TV Show", df["type"]))
    return df

# ==============
# SIDEBAR — Data & Filters
# ==============
st.sidebar.header("Data source")
st.sidebar.write("Sube un CSV combinado o usa tus archivos locales.")
uploaded_file = st.sidebar.file_uploader("Upload combined CSV", type=["csv"])

if uploaded_file is None:
    st.info("Sube un CSV con columnas mínimas: title, type, platform, release_year, duration, genres, country, description (rating opcional).")
    st.stop()

df = load_csv(uploaded_file)
df = ensure_platform_type(df)
df = parse_duration_cols(df)
df = build_genres(df)
df = build_country_primary(df)
df = normalize_rating(df)
df = build_decade(df)

# Available filters
platforms = sorted(df["platform"].dropna().unique().tolist()) if "platform" in df.columns else []
types     = sorted(df["type"].dropna().unique().tolist()) if "type" in df.columns else []
years     = df["release_year"].dropna().astype(int) if "release_year" in df.columns else pd.Series([], dtype=int)

st.sidebar.header("Filters")
sel_platform = st.sidebar.multiselect("Platform", platforms, default=platforms)
sel_type     = st.sidebar.multiselect("Type", types, default=types)

if len(years) > 0:
    min_y, max_y = int(years.min()), int(years.max())
    sel_years = st.sidebar.slider("Release year range", min_y, max_y, (min_y, max_y))
else:
    sel_years = (None, None)

# Optional filters
top_genres = df["primary_genre"].dropna().value_counts().index[:30].tolist() if "primary_genre" in df.columns else []
sel_genre  = st.sidebar.selectbox("Primary genre (optional)", ["(All)"] + top_genres, index=0)

top_countries = df["country_primary"].dropna().value_counts().index[:30].tolist() if "country_primary" in df.columns else []
sel_country   = st.sidebar.selectbox("Primary country (optional)", ["(All)"] + top_countries, index=0)

# Apply filters
mask = pd.Series(True, index=df.index)

if sel_platform:
    mask &= df["platform"].isin(sel_platform)

if sel_type:
    mask &= df["type"].isin(sel_type)

if sel_years[0] is not None:
    y = pd.to_numeric(df["release_year"], errors="coerce")
    mask &= (y >= sel_years[0]) & (y <= sel_years[1])

if sel_genre != "(All)" and "primary_genre" in df.columns:
    mask &= df["primary_genre"].eq(sel_genre)

if sel_country != "(All)" and "country_primary" in df.columns:
    mask &= df["country_primary"].eq(sel_country)

df_f = df[mask].copy()

# ==============
# LAYOUT
# ==============
st.title("Netflix vs Disney+ — Catalog Dashboard")
st.caption("Quick exploratory dashboard from the background analysis (EDA).")

# --- KPIs ---
c1, c2, c3, c4, c5 = st.columns(5)
total_titles = len(df_f)
movie_share  = (df_f["type"].eq("Movie").mean() if total_titles else 0.0)
median_rt    = float(np.nanmedian(df_f.loc[df_f["type"].eq("Movie"), "duration_min"])) if "duration_min" in df_f else np.nan
uniq_countries = df_f["country_primary"].nunique(dropna=True) if "country_primary" in df_f else 0
uniq_genres    = df_f["primary_genre"].nunique(dropna=True) if "primary_genre" in df_f else 0

c1.metric("Titles", f"{total_titles:,}")
c2.metric("Movie share", f"{movie_share:.0%}")
c3.metric("Median runtime (min)", "—" if np.isnan(median_rt) else f"{int(median_rt)}")
c4.metric("Countries (unique)", uniq_countries)
c5.metric("Primary genres (unique)", uniq_genres)

st.markdown("---")

# --- Ratings mix (stacked bar) ---
if "rating_norm" in df_f.columns and "platform" in df_f.columns:
    st.subheader("Ratings mix by platform")
    rating_tab = (df_f.assign(_one=1)
                    .pivot_table(index="platform", columns="rating_norm", values="_one", aggfunc="sum", fill_value=0))
    rating_share = rating_tab.div(rating_tab.sum(axis=1), axis=0).sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    bottom = np.zeros(len(rating_share))
    for col in ["G","Teen","Mature","Other","Unknown"]:
        if col in rating_share.columns:
            ax.bar(rating_share.index, rating_share[col].values, bottom=bottom, label=col)
            bottom += rating_share[col].values
    ax.set_ylabel("Share")
    ax.set_xlabel("")
    ax.set_title("Ratings mix (share of catalog)")
    ax.legend(title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)

st.markdown("---")

# --- Top genres & Top countries ---
colA, colB = st.columns(2)

with colA:
    st.subheader("Top primary genres")
    if "primary_genre" in df_f.columns:
        gcount = df_f["primary_genre"].value_counts().head(10).sort_values()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(gcount.index, gcount.values)
        ax.set_xlabel("Titles")
        st.pyplot(fig)
    else:
        st.info("No primary_genre available.")

with colB:
    st.subheader("Top countries (primary)")
    if "country_primary" in df_f.columns:
        ccount = df_f["country_primary"].value_counts().head(10).sort_values()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(ccount.index, ccount.values)
        ax.set_xlabel("Titles")
        st.pyplot(fig)
    else:
        st.info("No country_primary available.")

st.markdown("---")

# --- Distributions: runtime (movies) and seasons (TV) ---
colC, colD = st.columns(2)

with colC:
    st.subheader("Movie runtime distribution (min)")
    if "duration_min" in df_f.columns:
        m = df_f.loc[df_f["type"].eq("Movie"), "duration_min"].dropna()
        if len(m) > 0:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(m, bins=30)
            ax.set_xlabel("Minutes")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No movie runtimes available.")
    else:
        st.info("No duration_min available.")

with colD:
    st.subheader("TV seasons distribution")
    if "seasons_n" in df_f.columns:
        s = df_f.loc[df_f["type"].eq("TV Show"), "seasons_n"].dropna()
        if len(s) > 0:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(s, bins=np.arange(1, s.max()+2)-0.5)
            ax.set_xlabel("Seasons")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No seasons data available.")
    else:
        st.info("No seasons_n available.")

st.markdown("---")

# --- Searchable table ---
st.subheader("Catalog (filtered)")
q = st.text_input("Search in title/description (optional)")
df_show = df_f.copy()
if q:
    q_low = q.lower()
    cols = []
    if "title" in df_show.columns: cols.append("title")
    if "description" in df_show.columns: cols.append("description")
    if cols:
        mask_q = pd.Series(False, index=df_show.index)
        for c in cols:
            mask_q |= df_show[c].astype(str).str.lower().str.contains(q_low, na=False)
        df_show = df_show[mask_q]

cols_order = [c for c in ["title","type","platform","release_year","duration","duration_min","seasons_n","primary_genre","country_primary","rating_norm"] if c in df_show.columns]
st.dataframe(df_show[cols_order].reset_index(drop=True), use_container_width=True)
