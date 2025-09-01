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

st.set_page_config(
    page_title="Netflix vs Disney+ — Catalog Explorer",
    layout="wide",
)

# ------------------------
# Helpers & preprocessing
# ------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    nf = pd.read_csv("data/netflix_titles.csv")
    nf["platform"] = "Netflix"

    dp = pd.read_csv("data/disney_plus_titles.csv")
    dp["platform"] = "Disney+"

    df = pd.concat([nf, dp], ignore_index=True)

    # Normalize column names we depend on
    df.columns = [c.strip() for c in df.columns]
    # Some datasets have 'listed_in' (Netflix style); keep as-is and also mirror to 'genres'
    if "listed_in" in df.columns and "genres" not in df.columns:
        df["genres"] = df["listed_in"]

    # Safe minimum expected columns
    needed = {"title", "type", "platform", "release_year", "duration", "genres"}
    missing = needed.difference(df.columns)
    if missing:
        st.warning(f"Missing expected columns: {sorted(missing)} — the app will try to continue.")

    return df


def extract_minutes(x):
    if isinstance(x, str) and "min" in x:
        try:
            return pd.to_numeric(x.split()[0], errors="coerce")
        except Exception:
            return np.nan
    return np.nan


def extract_seasons(x):
    if isinstance(x, str) and "Season" in x:
        try:
            return pd.to_numeric(x.split()[0], errors="coerce")
        except Exception:
            return np.nan
    return np.nan


def primary_genre_from_listed(s):
    if pd.isna(s):
        return np.nan
    toks = [t.strip() for t in str(s).split(",") if t.strip()]
    return toks[0] if toks else np.nan


RATING_MAP = {
    # Kids / General
    "G": "G",
    "TV-G": "G",
    "TV-Y": "G",
    "TV-Y7": "G",
    "TV-PG": "G",
    "PG": "G",
    "PG-13": "Teen",
    "TV-14": "Teen",
    # Mature
    "R": "Mature",
    "NC-17": "Mature",
    "TV-MA": "Mature",
}
# ------------------------


# ----------
# Load data
# ----------
df_raw = load_data()

# Feature parsing (minutes/seasons)
df = df_raw.copy()
df["duration_min"] = df["duration"].apply(extract_minutes) if "duration" in df.columns else np.nan
df["seasons_n"] = df["duration"].apply(extract_seasons) if "duration" in df.columns else np.nan
df["primary_genre"] = df["genres"].apply(primary_genre_from_listed) if "genres" in df.columns else np.nan

# Ratings normalization (robust to NaNs; avoids fillna with ndarray)
if "rating" in df.columns:
    col = df["rating"]
    mapped = col.map(RATING_MAP)  # mapped to G/Teen/Mature or NaN
    # If original is NaN -> "Unknown"; else, if mapped is NaN -> "Other"
    rating_norm = np.where(col.isna(), "Unknown", mapped.fillna("Other"))
    df["rating_norm"] = pd.Series(rating_norm, index=df.index, dtype="object")
else:
    df["rating_norm"] = "Unknown"

# Canonical type cleanup (capitalize consistently)
if "type" in df.columns:
    df["type"] = df["type"].astype(str).str.strip().str.title()

# Filter only rows with platform + type present to avoid chart errors
df_f = df.dropna(subset=["platform", "type"], how="any")


# --------------
# Page header
# --------------
st.title("Netflix vs Disney+ — Catalog Explorer")
st.caption("Quick, reproducible view of the comparative EDA results. Data loaded from `/data` in this repo.")


# -------------------
# KPI summary up top
# -------------------
total_titles = len(df_f)
movie_share = (
    (df_f["type"].str.lower() == "movie").mean() if "type" in df_f.columns else np.nan
)

median_movie_runtime = (
    df_f.loc[df_f["type"].str.lower() == "movie", "duration_min"].median()
    if "duration_min" in df_f.columns and "type" in df_f.columns
    else np.nan
)

median_tv_seasons = (
    df_f.loc[df_f["type"].str.lower() == "tv show", "seasons_n"].median()
    if "seasons_n" in df_f.columns and "type" in df_f.columns
    else np.nan
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Titles", f"{total_titles:,.0f}")
c2.metric("Movies share", f"{movie_share*100:,.1f}%")
c3.metric("Median movie runtime", f"{median_movie_runtime:,.0f} min")
c4.metric("Median TV seasons", f"{median_tv_seasons:,.1f}")


# ----------
# Tabs
# ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["By platform & type", "Release trend", "Top genres", "Ratings mix"]
)

# ------------------------------
# By platform & type  (FIXED)
# ------------------------------
with tab1:
    st.subheader("Counts by platform and type")

    counts = (
        df_f.groupby(["platform", "type"])
            .size()
            .rename("count")
            .reset_index()  # Convert MultiIndex to columns (prevents KeyError)
    )

    # Pivot rows = platform, columns = type
    pivot = counts.pivot(index="platform", columns="type", values="count").fillna(0)

    # Keep nice column order if both exist
    desired_cols = [c for c in ["Movie", "Tv Show", "TV Show"] if c in pivot.columns]
    # Normalize TV Show column name if inconsistent casing
    if "Tv Show" in pivot.columns and "TV Show" not in pivot.columns:
        pivot = pivot.rename(columns={"Tv Show": "TV Show"})
        desired_cols = [c if c != "Tv Show" else "TV Show" for c in desired_cols]

    if desired_cols:
        pivot = pivot[desired_cols]

    st.bar_chart(pivot)


# ---------------
# Release trend
# ---------------
with tab2:
    st.subheader("Release trend")

    # Overall per year
    year_counts = (
        df_f.dropna(subset=["release_year"])
        .groupby("release_year")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("release_year")
    )

    # By platform per year
    plat_year = (
        df_f.dropna(subset=["release_year"])
        .groupby(["release_year", "platform"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["release_year", "platform"])
    )

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Total titles released by year")
        st.line_chart(
            year_counts.set_index("release_year")
        )

    with c2:
        st.caption("Titles released by year — Netflix vs Disney+")
        chart = (
            alt.Chart(plat_year)
            .mark_line(point=False)
            .encode(
                x=alt.X("release_year:Q", title="Year"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("platform:N", title="Platform"),
                tooltip=["release_year", "platform", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)


# ------------
# Top genres
# ------------
with tab3:
    st.subheader("Top-10 primary genres — counts by platform")

    # Compute top-10 overall primary genres first
    top10 = (
        df_f["primary_genre"]
        .dropna()
        .value_counts()
        .head(10)
        .index.tolist()
    )

    g_platform = (
        df_f[df_f["primary_genre"].isin(top10)]
        .groupby(["primary_genre", "platform"])
        .size()
        .rename("count")
        .reset_index()
    )

    # Pivot for bar chart (rows=primary_genre)
    g_pivot = (
        g_platform.pivot(index="primary_genre", columns="platform", values="count")
        .fillna(0)
        .sort_values(by=list(g_platform["platform"].unique()), ascending=False)
    )

    st.bar_chart(g_pivot)


# -------------
# Ratings mix
# -------------
with tab4:
    st.subheader("Ratings mix by platform (share of catalog)")

    share = (
        df_f.groupby(["platform", "rating_norm"])
        .size()
        .rename("count")
        .reset_index()
    )
    # Convert to shares per platform
    share["platform_total"] = share.groupby("platform")["count"].transform("sum")
    share["share"] = share["count"] / share["platform_total"]

    # Order ratings
    rating_order = ["G", "Teen", "Mature", "Other", "Unknown"]
    share["rating_norm"] = pd.Categorical(share["rating_norm"], categories=rating_order, ordered=True)

    stacked = (
        alt.Chart(share)
        .mark_bar()
        .encode(
            x=alt.X("platform:N", title="Platform"),
            y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share"),
            color=alt.Color("rating_norm:N", title="Rating"),
            tooltip=["platform", "rating_norm", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=360)
    )
    st.altair_chart(stacked, use_container_width=True)


# ===== Raw sample =====
st.markdown("### Sample (first 20 rows)")
st.dataframe(df_f.head(20), use_container_width=True)
