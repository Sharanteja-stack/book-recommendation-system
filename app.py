# app.py - BookFlix (Netflix-style Streamlit GUI)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import requests
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import os

# ---- Page config ----
st.set_page_config(page_title="BookFlix", layout="wide")

ARTIFACTS = Path("artifacts")
DATA_DIR = Path("data")
ARTIFACTS.mkdir(exist_ok=True)

BOOK_NAMES_PKL = ARTIFACTS / "book_name.pkl"
PIVOT_PKL = ARTIFACTS / "book_pivot.pkl"
FINAL_PKL = ARTIFACTS / "final_rating.pkl"
SIM_PKL = ARTIFACTS / "similarity.pkl"
MODEL_PKL = ARTIFACTS / "model.pkl"
BOOKS_CSV = DATA_DIR / "BX-Books.csv"
RATINGS_CSV = DATA_DIR / "BX-Book-Ratings.csv"

PLACEHOLDER = "https://via.placeholder.com/128x192.png?text=No+Cover"

# ---- Helper functions ----
@st.cache_data
def safe_load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def try_fetch_image(url):
    if not isinstance(url, str) or url.strip() == "":
        return None
    try:
        resp = requests.get(url, timeout=5)
        img = Image.open(BytesIO(resp.content))
        return img
    except Exception:
        return None

def recommend_from_similarity(title, sim, pivot, k=5):
    if sim is None:
        return []
    titles = list(pivot.columns)
    if title not in titles:
        return []
    idx = titles.index(title)
    distances = list(enumerate(sim[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    recs = [titles[i] for i, _ in distances[1:k + 1]]
    return recs

def recommend_with_knn(title, model, pivot, k=5):
    titles = list(pivot.columns)
    if title not in titles:
        return []
    idx = titles.index(title)
    try:
        distances, indices = model.kneighbors(pivot.T.iloc[idx:idx + 1, :], n_neighbors=k + 1)
        recs = [titles[i] for i in indices[0] if i != idx]
        return recs[:k]
    except Exception:
        return []

# ---- Load Artifacts ----
book_names = safe_load_pickle(BOOK_NAMES_PKL)
pivot = safe_load_pickle(PIVOT_PKL)
final_rating = safe_load_pickle(FINAL_PKL)
sim_matrix = safe_load_pickle(SIM_PKL)
model = safe_load_pickle(MODEL_PKL)

# If still missing, show error message
if book_names is None or pivot is None:
    st.title("BookFlix - Book Recommendation App")
    st.error("""
    Artifacts not found.
    Please place your processed files inside the `artifacts/` folder:
      - book_name.pkl
      - book_pivot.pkl
      - final_rating.pkl
    Or place raw CSVs in `data/` folder:
      - BX-Books.csv
      - BX-Book-Ratings.csv
    """)
    st.stop()

# ---- UI ----
st.markdown("<h1 style='text-align:center;'>📚 BookFlix</h1>", unsafe_allow_html=True)
st.markdown("### A Netflix-style Book Recommendation App")

with st.sidebar:
    st.title("Controls")
    selected = st.selectbox("Choose a book:", sorted(book_names))
    k = st.slider("Number of recommendations:", 1, 10, 5)

st.subheader(f"Recommendations for: {selected}")

if model is not None:
    recs = recommend_with_knn(selected, model, pivot, k)
elif sim_matrix is not None:
    recs = recommend_from_similarity(selected, sim_matrix, pivot, k)
else:
    recs = []

if not recs:
    st.warning("No recommendations found.")
else:
    cols = st.columns(k)
    for i, t in enumerate(recs):
        c = cols[i]
        c.image(PLACEHOLDER)
        c.markdown(f"**{t}**")

st.markdown("---")
st.caption("Built with ❤️ by Sharan using Streamlit")
