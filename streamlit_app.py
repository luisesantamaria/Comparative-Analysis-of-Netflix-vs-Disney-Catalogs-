# streamlit_app.py
# ------------------------------------------------------------
# Netflix vs Disney+ — Catalog Explorer (Streamlit)
# Data are loaded from /data in this repository:
#   - data/netflix_titles.csv
#   - data/disney_plus_titles.csv
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re

APP_VERSION = "v3.3-ecdf+genre-normalization"

st.set_page_config(
    page_title="Netflix vs Disney+ — Catalog Explorer",
    layout="wide",
)

# ------------------------
# Helpers & preprocessing
# ------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    nf = pd.read_csv("data/netflix_titles.csv", encoding="utf-8", low_memory=False)
    nf["platform"] = "Netflix"

    dp = pd.read_csv("data/disney_plus_titles.csv", encoding="utf-8", low_memory=False)
    dp["platform"] = "Disney+"

    df = pd.concat([nf, dp], ignore_index=True)

    # Column normalizations
    df.columns = [c.strip() for c in df.columns]
    if "listed_in" in df.columns and "genres" not in df.columns:
        df["genres"] = df["listed_in"]

    return df


def extract_minutes(x: str):
    if isinstance(x, str) and "min" in x:
        m = re.search(r"(\d+)\s*min", x)
        if m:
            return pd.to_numeric(m.group(1), errors="coerce")
    return np.nan


def extract_seasons(x: str):
    if isinstance(x, str) and "Season" in x:
        m = re.search(r"(\d+)\s*Season", x, flags=re.IGNORECASE)
        if m:
            return pd.to_numeric(m.group(1), errors="coerce")
    return np.nan


def split_genres(s: str):
    if pd.isna(s):
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]


def primary_genre(s: str):
    toks = split_genres(s)
    return toks[0] if toks else np.nan


def country_primary(s: str):
    if pd.isna(s):
        return np.nan
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return parts[0] if parts else np.nan


def normalize_genre_token(g):
    """
    Unifica variantes de género para que no salgan barras duplicadas.
    - Action & Adventure: maneja -, –, —, '&', 'and', y espacios.
    - Comedy/Comedies: colapsa a 'Comedy' (NO tocar 'Stand-Up Comedy').
    """
    if pd.isna(g):
        return np.nan

    t = str(g).strip()
    low = (
        t.lower()
        .replace("—", "-")
        .replace("–", "-")
    )
    # tratar &, 'and' y guiones como equivalentes
    low = re.sub(r"\band\b", "&", low)
    low = re.sub(r"\s*-\s*", " & ", low)   # guiones -> &
    low = re.sub(r"\s*&\s*", " & ", low)   # normalizar & con espacios
    low = re.sub(r"\s+", " ", low).strip()

    # 1) Stand-Up Comedy se conserva
    if "stand-up" in low and "comedy" in low:
        return "Stand-Up Comedy"

    # 2) Action & Adventure
    if re.search(r"\baction\b", low) and re.search(r"\badventure\b", low):
        return "Action & Adventure"

    # 3) Comedy / Comedies
    if re.search(r"\bcomed(y|ies)\b", low):
        return "Comedy"

    # 4) Default: Title Case
    return t.title()


RATING_MAP = {
    # Kids / General
    "G": "G", "TV-G": "G", "TV-Y": "G", "TV-Y7": "G", "TV-PG": "G", "PG": "G",
    # Teen
    "PG-13": "Teen", "TV-14": "Teen",
    # Mature
    "R": "Mature", "NC-17": "Mature", "TV-MA": "Mature",
}

def normalize_rating(df: pd.DataFrame) -> pd.Series:
    """Robusta y compatible con Py3.9; evita fillna con ndarray."""
    if "rating" not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")

    raw = df["rating"]
    col = raw.astype(str).str.upper().str.strip()
    mapped = col.map(RATING_MAP)  # puede contener NaN

    out = pd.Series("Other", index=df.index, dtype="object")
    out[raw.isna()] = "Unknown"              # NaN originales -> Unknown
    known_mask = mapped.notna()
    out[known_mask] = mapped[known_mask]     # mapeos conocidos
    tbg_mask = col.str.contains("TBG", na=False) & out.eq("Other")
    out[tbg_mask] = "Teen"                   # códigos tipo TBG -> Teen

    return out


@st.cache_data(show_spinner=False)
def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Duration-derived
    if "duration" in df.columns:
        df["duration_min"] = df["duration"].apply(extract_minutes)
        df["seasons_n"] = df["duration"].apply(extract_seasons)

    # Clean type
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.title().replace({"Tv Show": "TV Show"})

    # Primary genre (normalizado)
    if "genres" in df.columns:
        df["primary_genre"] = df["genres"].apply(primary_genre).apply(normalize_genre_token)

    # Country primary
    if "country" in df.columns:
        df["country_primary"] = df["country"].apply(country_primary)

    # Ratings normalized
    df["rating_norm"] = normalize_rating(df)

    # Year & decade
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df = df[df["release_year"].between(1900, 2035)]
        df["decade"] = (df["release_year"] // 10) * 10

    return df


# ----------
# Load + FE
# ----------
df_raw = load_data()
df = prepare_features(df_raw)

# -------------------
# Sidebar: Filters + cache control
# -------------------
st.sidebar.header("Filters")

if st.sidebar.button("Clear cache & reload"):
    st.cache_data.clear()
    st.rerun()

platforms = sorted(df["platform"].dropna().unique().tolist()) if "platform" in df.columns else []
types = sorted(df["type"].dropna().unique().tolist()) if "type" in df.columns else []
years = df["release_year"].dropna().astype(int) if "release_year" in df.columns else pd.Series([], dtype=int)

sel_platform = st.sidebar.multiselect("Platform", platforms, default=platforms)
sel_type = st.sidebar.multiselect("Type", types, default=types)
if len(years) > 0:
    min_y, max_y = int(years.min()), int(years.max())
    sel_years = st.sidebar.slider("Release year range", min_y, max_y, (min_y, max_y))
else:
    sel_years = (None, None)

# Optional filters
genres_all = df["primary_genre"].dropna().value_counts().index[:40].tolist() if "primary_genre" in df.columns else []
sel_genre = st.sidebar.selectbox("Primary genre (optional)", ["(All)"] + genres_all, index=0)

ratings_all = ["G", "Teen", "Mature", "Other", "Unknown"]
sel_ratings = st.sidebar.multiselect("Rating category (optional)", ratings_all, default=ratings_all)

q = st.sidebar.text_input("Search title/description (optional)")

# Apply filters
mask = pd.Series(True, index=df.index)
if sel_platform:
    mask &= df["platform"].isin(sel_platform)
if sel_type:
    mask &= df["type"].isin(sel_type)
if sel_years[0] is not None:
    mask &= df["release_year"].between(sel_years[0], sel_years[1])
if sel_genre != "(All)" and "primary_genre" in df.columns:
    mask &= df["primary_genre"].eq(sel_genre)
if sel_ratings and "rating_norm" in df.columns:
    mask &= df["rating_norm"].isin(sel_ratings)
if q:
    q_low = q.lower()
    cols = []
    if "title" in df.columns: cols.append("title")
    if "description" in df.columns: cols.append("description")
    if cols:
        contains = pd.Series(False, index=df.index)
        for c in cols:
            contains |= df[c].astype(str).str.lower().str.contains(q_low, na=False)
        mask &= contains

df_f = df[mask].copy()

# --------------
# Header + KPIs
# --------------
st.title("Netflix vs Disney+ — Catalog Explorer")
st.caption(f"Interactive EDA. Data loaded from `/data` — Build {APP_VERSION}")

total_titles = len(df_f)
movie_share = (df_f["type"].eq("Movie").mean()*100) if "type" in df_f.columns else np.nan
median_movie_runtime = df_f.loc[df_f["type"].eq("Movie"), "duration_min"].median() if "duration_min" in df_f.columns else np.nan
median_tv_seasons = df_f.loc[df_f["type"].eq("TV Show"), "seasons_n"].median() if "seasons_n" in df_f.columns else np.nan
unique_genres = df_f["primary_genre"].nunique(dropna=True) if "primary_genre" in df_f.columns else 0
unique_countries = df_f["country_primary"].nunique(dropna=True) if "country_primary" in df_f.columns else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Titles", f"{total_titles:,}")
k2.metric("Movies share", f"{movie_share:.1f}%")
k3.metric("Median movie runtime", "—" if pd.isna(median_movie_runtime) else f"{median_movie_runtime:.0f} min")
k4.metric("Median TV seasons", "—" if pd.isna(median_tv_seasons) else f"{median_tv_seasons:.1f}")
k5.metric("Primary genres", unique_genres)
k6.metric("Countries (primary)", unique_countries)

st.markdown("---")

# ----------
# Tabs
# ----------
tab_over, tab_pt, tab_trend, tab_rt, tab_gen, tab_cty, tab_rate, tab_deep, tab_data = st.tabs(
    ["Overview", "Platform & Type", "Release Trend", "Runtime & Seasons", "Genres", "Countries", "Ratings", "Deep Dives", "Data"]
)

# ----------------
# Overview
# ----------------
with tab_over:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Catalog by platform")
        plat_counts = df_f.groupby("platform").size().rename("count").reset_index()
        ch = (
            alt.Chart(plat_counts)
            .mark_bar()
            .encode(
                x=alt.X("platform:N", title="Platform"),
                y=alt.Y("count:Q", title="Titles"),
                color=alt.Color("platform:N", legend=None),
                tooltip=["platform", "count"]
            )
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)

    with col2:
        st.subheader("Movies vs TV — 100% stacked by platform")
        base = df_f.groupby(["platform", "type"]).size().rename("count").reset_index()
        if not base.empty:
            base["total"] = base.groupby("platform")["count"].transform("sum")
            base["share"] = base["count"] / base["total"]
            ch2 = (
                alt.Chart(base)
                .mark_bar()
                .encode(
                    x=alt.X("platform:N", title="Platform"),
                    y=alt.Y("share:Q", title="Share", axis=alt.Axis(format="%")),
                    color=alt.Color("type:N", title="Type"),
                    tooltip=["platform","type", alt.Tooltip("share:Q", format=".1%")]
                )
                .properties(height=320)
            )
            st.altair_chart(ch2, use_container_width=True)
        else:
            st.info("No data for current filter.")

# ------------------------------
# Platform & Type
# ------------------------------
with tab_pt:
    st.subheader("Counts by platform and type")
    counts = df_f.groupby(["platform", "type"]).size().rename("count").reset_index()
    if counts.empty:
        st.info("No data for current filter.")
    else:
        ch = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("platform:N", title="Platform"),
                y=alt.Y("count:Q", title="Titles"),
                color=alt.Color("type:N", title="Type"),
                column=alt.Column("type:N", title=""),
                tooltip=["platform","type","count"]
            )
            .resolve_scale(y="shared")
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)

# ----------------
# Release Trend
# ----------------
with tab_trend:
    st.subheader("Titles released by year — totals and splits")
    base = df_f.dropna(subset=["release_year"])

    if base.empty:
        st.info("No release_year available for current filter.")
    else:
        # Total by year
        total_year = base.groupby("release_year").size().rename("count").reset_index()

        # Cumulative total
        cum_year = total_year.copy()
        cum_year["cumulative"] = cum_year["count"].cumsum()

        # By platform per year
        by_plat = base.groupby(["release_year", "platform"]).size().rename("count").reset_index()

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Total titles by year")
            ch_tot = (
                alt.Chart(total_year)
                .mark_line()
                .encode(
                    x=alt.X("release_year:Q", title="Year"),
                    y=alt.Y("count:Q", title="Total titles"),
                    tooltip=["release_year","count"]
                )
                .properties(height=320)
            )
            st.altair_chart(ch_tot, use_container_width=True)

        with c2:
            st.caption("Cumulative titles by year")
            ch_cum = (
                alt.Chart(cum_year)
                .mark_line()
                .encode(
                    x=alt.X("release_year:Q", title="Year"),
                    y=alt.Y("cumulative:Q", title="Cumulative total"),
                    tooltip=["release_year","cumulative"]
                )
                .properties(height=320)
            )
            st.altair_chart(ch_cum, use_container_width=True)

        st.caption("Titles by year — Netflix vs Disney+")
        ch_pl = (
            alt.Chart(by_plat)
            .mark_line()
            .encode(
                x=alt.X("release_year:Q", title="Year"),
                y=alt.Y("count:Q", title="Titles"),
                color=alt.Color("platform:N", title="Platform"),
                tooltip=["release_year","platform","count"]
            )
            .properties(height=320)
        )
        st.altair_chart(ch_pl, use_container_width=True)

# ------------------------
# Runtime & Seasons
# ------------------------
with tab_rt:
    st.subheader("Distributions — runtime (movies) & seasons (TV) by platform")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Movie runtime (minutes) — histogram by platform")
        mm = df_f.loc[df_f["type"].eq("Movie"), ["platform","duration_min"]].dropna()
        if mm.empty:
            st.info("No movie runtimes available.")
        else:
            ch = (
                alt.Chart(mm)
                .transform_filter(alt.datum.duration_min > 0)
                .mark_bar(opacity=0.6)
                .encode(
                    x=alt.X("duration_min:Q", bin=alt.Bin(maxbins=40), title="Minutes"),
                    y=alt.Y("count()", title="Count"),
                    color=alt.Color("platform:N", title="Platform"),
                    tooltip=[alt.Tooltip("count()", title="Count")]
                )
                .properties(height=320)
            )
            st.altair_chart(ch, use_container_width=True)

    with c2:
        st.caption("TV seasons — histogram by platform")
        ss = df_f.loc[df_f["type"].eq("TV Show"), ["platform","seasons_n"]].dropna()
        if ss.empty:
            st.info("No seasons data available.")
        else:
            ch2 = (
                alt.Chart(ss)
                .transform_filter(alt.datum.seasons_n >= 0)
                .mark_bar(opacity=0.6)
                .encode(
                    x=alt.X("seasons_n:Q", bin=alt.Bin(step=1), title="Seasons"),
                    y=alt.Y("count()", title="Count"),
                    color=alt.Color("platform:N", title="Platform"),
                    tooltip=[alt.Tooltip("count()", title="Count")]
                )
                .properties(height=320)
            )
            st.altair_chart(ch2, use_container_width=True)

    st.markdown("—")

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Movie runtime — boxplot by platform")
        if 'mm' in locals() and not mm.empty:
            ch_box = (
                alt.Chart(mm)
                .mark_boxplot()
                .encode(
                    x=alt.X("platform:N", title="Platform"),
                    y=alt.Y("duration_min:Q", title="Minutes"),
                    color=alt.Color("platform:N", legend=None)
                )
                .properties(height=320)
            )
            st.altair_chart(ch_box, use_container_width=True)

    with c4:
        st.caption("Movie runtime — ECDF by platform")
        if 'mm' in locals() and not mm.empty:
            # ECDF FIX: acumulado y total por plataforma
            ch_ecdf = (
                alt.Chart(mm)
                .transform_filter(alt.datum.duration_min > 0)
                .transform_window(
                    cumulative_count='count(*)',
                    sort=[{"field": "duration_min"}],
                    groupby=["platform"]
                )
                .transform_joinaggregate(
                    total='count(*)',
                    groupby=["platform"]
                )
                .transform_calculate(
                    ecdf="datum.cumulative_count / datum.total"
                )
                .mark_line()
                .encode(
                    x=alt.X("duration_min:Q", title="Minutes"),
                    y=alt.Y("ecdf:Q", title="ECDF", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0,1])),
                    color=alt.Color("platform:N", title="Platform"),
                    tooltip=[
                        alt.Tooltip("platform:N", title="Platform"),
                        alt.Tooltip("duration_min:Q", title="Minutes", format=".0f"),
                        alt.Tooltip("ecdf:Q", title="ECDF", format=".1%")
                    ]
                )
                .properties(height=320)
            )
            st.altair_chart(ch_ecdf, use_container_width=True)

# -----------
# Genres
# -----------
with tab_gen:
    st.subheader("Top-15 primary genres — by platform")
    if "primary_genre" not in df_f.columns or df_f["primary_genre"].dropna().empty:
        st.info("No primary_genre available.")
    else:
        top15 = df_f["primary_genre"].dropna().value_counts().head(15).index.tolist()
        sub = df_f[df_f["primary_genre"].isin(top15)]
        g = sub.groupby(["primary_genre","platform"]).size().rename("count").reset_index()

        ch = (
            alt.Chart(g)
            .mark_bar()
            .encode(
                y=alt.Y("primary_genre:N", sort="-x", title="Primary genre"),
                x=alt.X("count:Q", title="Titles"),
                color=alt.Color("platform:N", title="Platform"),
                tooltip=["primary_genre","platform","count"]
            )
            .properties(height=480)
        )
        st.altair_chart(ch, use_container_width=True)

        # Relative lift per platform (top-10)
        st.caption("Relative lift (over-index) by platform — top-10")
        overall = df_f["primary_genre"].value_counts()
        total = len(df_f)
        p_gen = overall / total
        counts_plat = df_f["platform"].value_counts()
        data_lift = []
        for plat in counts_plat.index:
            tmp = sub[sub["platform"] == plat]["primary_genre"].value_counts()
            p_gp = tmp / counts_plat[plat]
            lift = (p_gp / p_gen).replace([np.inf, -np.inf], np.nan).dropna()
            lift = lift.sort_values(ascending=False).head(10)
            data_lift.append(lift.to_frame(name="lift").assign(platform=plat).reset_index().rename(columns={"index":"primary_genre"}))
        lift_df = pd.concat(data_lift, ignore_index=True) if data_lift else pd.DataFrame(columns=["primary_genre","lift","platform"])

        if not lift_df.empty:
            ch_lift = (
                alt.Chart(lift_df)
                .mark_point(filled=True, size=90)
                .encode(
                    x=alt.X("lift:Q", title="Lift (P(genre|platform)/P(genre))"),
                    y=alt.Y("primary_genre:N", sort="-x", title="Primary genre"),
                    color=alt.Color("platform:N", title="Platform"),
                    tooltip=["platform","primary_genre", alt.Tooltip("lift:Q", format=".2f")]
                )
                .properties(height=480)
            )
            st.altair_chart(ch_lift, use_container_width=True)

# -------------
# Countries
# -------------
with tab_cty:
    st.subheader("Top-15 countries (primary) — by platform")
    if "country_primary" not in df_f.columns or df_f["country_primary"].dropna().empty:
        st.info("No country_primary available.")
    else:
        top15c = df_f["country_primary"].dropna().value_counts().head(15).index.tolist()
        subc = df_f[df_f["country_primary"].isin(top15c)]
        c = subc.groupby(["country_primary","platform"]).size().rename("count").reset_index()
        ch = (
            alt.Chart(c)
            .mark_bar()
            .encode(
                y=alt.Y("country_primary:N", sort="-x", title="Country"),
                x=alt.X("count:Q", title="Titles"),
                color=alt.Color("platform:N", title="Platform"),
                tooltip=["country_primary","platform","count"]
            )
            .properties(height=480)
        )
        st.altair_chart(ch, use_container_width=True)

# -----------
# Ratings
# -----------
with tab_rate:
    st.subheader("Ratings mix — share and counts")
    if "rating_norm" not in df_f.columns:
        st.info("No rating information available.")
    else:
        base = df_f.groupby(["platform","rating_norm"]).size().rename("count").reset_index()
        if base.empty:
            st.info("No data for current filter.")
        else:
            base["platform_total"] = base.groupby("platform")["count"].transform("sum")
            base["share"] = base["count"] / base["platform_total"]
            order = ["G", "Teen", "Mature", "Other", "Unknown"]
            base["rating_norm"] = pd.Categorical(base["rating_norm"], categories=order, ordered=True)

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Share of catalog by rating")
                ch_share = (
                    alt.Chart(base)
                    .mark_bar()
                    .encode(
                        x=alt.X("platform:N", title="Platform"),
                        y=alt.Y("share:Q", title="Share", axis=alt.Axis(format="%")),
                        color=alt.Color("rating_norm:N", title="Rating"),
                        tooltip=["platform","rating_norm", alt.Tooltip("share:Q", format=".1%")]
                    )
                    .properties(height=360)
                )
                st.altair_chart(ch_share, use_container_width=True)

            with c2:
                st.caption("Counts by rating")
                ch_cnt = (
                    alt.Chart(base)
                    .mark_bar()
                    .encode(
                        x=alt.X("platform:N", title="Platform"),
                        y=alt.Y("count:Q", title="Count"),
                        color=alt.Color("rating_norm:N", title="Rating"),
                        tooltip=["platform","rating_norm","count"]
                    )
                    .properties(height=360)
                )
                st.altair_chart(ch_cnt, use_container_width=True)

# -----------
# Deep Dives
# -----------
with tab_deep:
    st.subheader("Runtime median by decade and platform (movies)")
    movies = df_f[df_f["type"].eq("Movie")].dropna(subset=["duration_min"])
    if not movies.empty:
        med_dec = (
            movies.groupby(["decade","platform"])["duration_min"]
                  .median().reset_index(name="median_min")
        )
        ch_med = (
            alt.Chart(med_dec)
            .mark_line(point=True)
            .encode(
                x=alt.X("decade:Q", title="Decade"),
                y=alt.Y("median_min:Q", title="Median runtime (min)"),
                color=alt.Color("platform:N", title="Platform"),
                tooltip=["decade","platform", alt.Tooltip("median_min:Q", format=".0f")]
            )
            .properties(height=360)
        )
        st.altair_chart(ch_med, use_container_width=True)
    else:
        st.info("No movie runtime data available for current filter.")

# -----------
# Data (descarga y tabla)
# -----------
with tab_data:
    st.subheader("Data")

    st.dataframe(df, use_container_width=True, height=500)

    # Descargas
    st.markdown("**Downloads**")
    # 1) CSV de la base combinada completa
    csv_full = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Data Catalog",
        data=csv_full,
        file_name="catalog_combined_full.csv",
        mime="text/csv",
        use_container_width=True
    )
