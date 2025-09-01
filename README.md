# Netflix vs Disney+ — Catalog Analysis

[![Launch App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://netflixvsdisneycataloganalysis.streamlit.app/)

This project provides a structured exploratory data analysis (EDA) of Netflix and Disney+ catalogs.  
It covers cleaning, feature engineering, distributional comparisons, genre and country diversity, maturity ratings, and topic extraction using TF-IDF. 

---

## Live Dashboard
Explore the interactive dashboard here:  
👉 [[netflixvsdisneyanalysis.streamlit.app](https://netflixvsdisneycataloganalysis.streamlit.app/)](https://netflixvsdisneycataloganalysis.streamlit.app/)

---

## Highlights
- Full pipeline: from raw catalog data → cleaning → feature engineering → visual analysis.
- Platform comparisons: runtime, release trends, genres, countries, and ratings.
- TF-IDF topic extraction to identify common themes.
- Executive conclusions with strategic insights.

---

## Repository structure
- `notebook.ipynb` — main Jupyter/Colab notebook with analysis (run-all successful, outputs included).
- `report.pdf` — formatted PDF background report (to be added).
- `requirements.txt` — minimal dependencies (to be added).
- `assets/` — key charts for documentation (to be added).

---

## Data
The analysis uses public catalog snapshots.  
Expected schema: `title, type, platform, release_year, duration, genres, country, description (optional: rating)`.

---

## How to run locally
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
