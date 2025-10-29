# app.py - BookFlix (Netflix-style Streamlit GUI)
# Generated to work with your Colab notebook artifacts.

"""
Requirements:
- streamlit
- pandas
- numpy
...



import os
import pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import io
import requests

# -------------------- Configuration --------------------
st.set_page_config(page_title="BookFlix", layout="wide")
ARTIFACTS_DIR = Path("./artifacts")
DATA_DIR = Path("./data")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PKL = ARTIFACTS_DIR / "model.pkl"
BOOK_PIVOT_PKL = ARTIFACTS_DIR / "book_pivot.pkl"
FINAL_RATING_PKL = ARTIFACTS_DIR / "final_rating.pkl"
BOOK_NAME_PKL = ARTIFACTS_DIR / "book_name.pkl"

# Expected CSV filenames (place them under ./data)
CSV_BOOKS = DATA_DIR / "BX-Books.csv"
CSV_RATINGS = DATA_DIR / "BX-Book-Ratings.csv"

PLACEHOLDER = "https://via.placeholder.com/128x192.png?text=No+Cover"

# -------------------- Utilities --------------------

def safe_load_pickle(p):
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle from {p}: {e}")
        return None


def save_pickle(obj, p):
    with open(p, 'wb') as f:
        pickle.dump(obj, f)


def try_fetch_image(img_url, max_size=(300, 450)):
    """Try to fetch image from URL and return PIL image or None."""
    if not isinstance(img_url, str) or not img_url:
        return None
    try:
        resp = requests.get(img_url, timeout=6)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.thumbnail(max_size)
        return img
    except Exception:
        return None

# -------------------- Model build --------------------

@st.cache_data
def build_model_from_csv(books_csv, ratings_csv, threshold_book_ratings=20):
    st.info("Artifacts not found. Attempting to build from CSVs...")
    # Read using semicolon separator (common for BX dataset)
    try:
        books = pd.read_csv(books_csv, sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        st.error(f"Error reading books CSV: {e}")
        return None, None, None
    try:
        ratings = pd.read_csv(ratings_csv, sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        st.error(f"Error reading ratings CSV: {e}")
        return None, None, None

    # Normalise column names
    books.columns = [c.strip() for c in books.columns]
    ratings.columns = [c.strip() for c in ratings.columns]

    # Find title col
    title_col = None
    for c in books.columns:
        if 'title' in c.lower():
            title_col = c
            break
    if title_col is None:
        st.error("Could not find book title column in books CSV.")
        return None, None, None

    # rating columns
    user_col = None
    rating_col = None
    for c in ratings.columns:
        lc = c.lower()
        if 'user' in lc:
            user_col = c
        if 'rating' in lc:
            rating_col = c
    if user_col is None:
        st.error("Could not find user ID column in ratings CSV.")
        return None, None, None
    if rating_col is None:
        st.error("Could not find rating column in ratings CSV.")
        return None, None, None


    # merge on ISBN if present
    join_b = None
    join_r = None
    for c in books.columns:
        if 'isbn' in c.lower():
            join_b = c
            break
    for c in ratings.columns:
        if 'isbn' in c.lower() or 'isbn' in c.lower():
            join_r = c
            break

    merged = ratings.copy()
    if join_b and join_r:
        merged = ratings.merge(books[[join_b, title_col]], left_on=join_r, right_on=join_b, how='left')
        merged.rename(columns={title_col: 'title'}, inplace=True)
    elif 'title' in ratings.columns:
        merged = ratings.rename(columns={'title':'title'})
    else:
        st.error("Could not merge books and ratings on ISBN or find title in ratings.")
        return None, None, None


    merged = merged[merged['title'].notnull()]

    merged['rating'] = merged[rating_col]

    final_rating = merged.groupby('title').agg({'rating':['count','mean']})
    final_rating.columns = ['num_rating','avg_rating']
    final_rating = final_rating.reset_index().rename(columns={'title':'title'})

    # pivot
    pivot = merged.pivot_table(index='title', columns='user_id', values='rating', aggfunc='mean').fillna(0)

    # Fit a NearestNeighbors model
    model = None # Default to None
    try:
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(pivot.values)
    except Exception as e:
        st.warning(f"Could not fit NearestNeighbors model: {e}")


    # Save artifacts
    try:
        save_pickle(list(pivot.index), BOOK_NAME_PKL) # Use pivot index for consistent book names
        save_pickle(pivot, BOOK_PIVOT_PKL)
        save_pickle(final_rating, FINAL_RATING_PKL)
        if model is not None:
            save_pickle(model, MODEL_PKL)
        st.success("Artifacts built and saved successfully!")
    except Exception as e:
        st.warning(f"Could not save artifacts: {e}")

    return model, pivot, final_rating


# -------------------- Main App Logic --------------------

st.markdown("___")
st.markdown("### Loading Data and Model...")

model = None
book_pivot = None
final_rating = None
book_list = None # Use book_list for selectbox options

# Try load artifacts
model = safe_load_pickle(MODEL_PKL)
book_pivot = safe_load_pickle(BOOK_PIVOT_PKL)
final_rating = safe_load_pickle(FINAL_RATING_PKL)
book_list = safe_load_pickle(BOOK_NAME_PKL) # Load book names for selectbox


if model is not None and book_pivot is not None and book_list is not None:
    st.success("Artifacts loaded successfully!")
else:
    st.warning("Artifacts not found or incomplete. Attempting to build from CSVs...")
    # Else build if CSVs available
    if CSV_BOOKS.exists() and CSV_RATINGS.exists():
        model, book_pivot, final_rating = build_model_from_csv(CSV_BOOKS, CSV_RATINGS)
        if book_pivot is not None:
            book_list = list(book_pivot.index) # Get book names from pivot if built
            st.success("Building from CSVs successful!")
        else:
            st.error("Building from CSVs failed.")
    else:
        st.error("CSV files not found in ./data/. Cannot build artifacts.")


# Check if critical artifacts are loaded after loading or building
if book_pivot is None or model is None or book_list is None:
    st.error("Could not load or build necessary data and model. Please check logs above.")
    st.stop()


# Try to load BX-Books.csv to extract metadata (author, images)
books_meta = None
if CSV_BOOKS.exists():
    try:
        books_meta = pd.read_csv(CSV_BOOKS, sep=';', encoding='latin-1', low_memory=False)
        books_meta.columns = [c.strip() for c in books_meta.columns]
    except Exception:
        try:
            books_meta = pd.read_csv(CSV_BOOKS, encoding='latin-1', low_memory=False)
            books_meta.columns = [c.strip() for c in books_meta.columns]
        except Exception:
            books_meta = None

# Helper to get metadata row for a title
def get_meta_for_title(title):
    if books_meta is None:
        return {"author":"Unknown","image":PLACEHOLDER}
    # expected columns
    title_col = None
    author_col = None
    image_col = None
    for c in books_meta.columns:
        lc = c.lower()
        if 'title' in lc:
            title_col = c
        if 'author' in lc:
            author_col = c
        if 'image' in lc or 'url' in lc:
            image_col = c
    if title_col is None:
        return {"author":"Unknown","image":PLACEHOLDE R}
    row = books_meta[books_meta[title_col].astype(str).str.strip().str.lower() == title.strip().lower()]
    if row.shape[0] == 0:
        # try fuzzy match by substring
        row = books_meta[books_meta[title_col].astype(str).str.lower().str.contains(title.strip().lower())]
    if row.shape[0] == 0:
        return {"author":"Unknown","image":PLACEHOLDER}
    r = row.iloc[0]
    author = r[author_col] if author_col and pd.notnull(r[author_col]) else "Unknown"
    img = r[image_col] if image_col and pd.notnull(r[image_col]) else PLACEHOLDER
    return {"author":str(author), "image":img}

# ----- UI -----
st.markdown("<h1 style='text-align:center'>📚 BookFlix</h1>", unsafe_allow_html=True)
st.markdown("#### A Netflix-style interface for your Book Recommendation System")

# Sidebar
with st.sidebar:
    st.title("Controls")
    q = st.text_input("Search books or authors")
    rec_k = st.slider("Recs per row", 1, 10, 5)
    show_top = st.checkbox("Show Top Rated", value=True)
    show_trending = st.checkbox("Show Trending", value=True)
    selected = st.selectbox("Pick a book for recommendations", options=sorted(book_list)) # Use book_list here


# Search results
st.subheader("Search")
if q:
    matches = [t for t in book_list if q.lower() in t.lower()]
    if len(matches) == 0:
        st.info("No matches found")
    else:
        cols = st.columns(5)
        for i, t in enumerate(matches[:20]):
            c = cols[i % 5]
            meta = get_meta_for_title(t)
            img = try_fetch_image(meta['image'])
            if img is not None:
                c.image(img, use_column_width=True)
            else:
                c.image(meta['image'] if isinstance(meta['image'], str) else PLACEHOLDER, use_column_width=True)
            c.markdown(f"**{t}**")
            c.caption(meta['author'])

st.markdown("---")

# Top Rated row
if show_top and final_rating is not None:
    st.subheader("Top Rated")
    # Ensure final_rating has necessary columns
    if isinstance(final_rating, pd.DataFrame) and 'avg_rating' in final_rating.columns and 'num_rating' in final_rating.columns:
        top_df = final_rating.sort_values(['avg_rating','num_rating'], ascending=False).head(20)
        cols = st.columns(5)
        for i, t in enumerate(top_df['title'].tolist()[:15]):
            c = cols[i % 5]
            meta = get_meta_for_title(t)
            img = try_fetch_image(meta['image'])
            if img is not None:
                c.image(img, use_column_width=True)
            else:
                c.image(meta['image'] if isinstance(meta['image'], str) else PLACEHOLDER, use_column_width=True)
            c.markdown(f"**{t}**")
            c.caption(meta['author'])
    else:
        st.warning("Could not display Top Rated books (final_rating data incomplete).")


# Trending row
if show_trending and final_rating is not None:
    st.subheader("Trending")
    # Ensure final_rating has necessary columns
    if isinstance(final_rating, pd.DataFrame) and 'num_rating' in final_rating.columns:
        trending = final_rating.sort_values('num_rating', ascending=False).head(20)
        cols = st.columns(5)
        for i, t in enumerate(trending['title'].tolist()[:15]):
            c = cols[i % 5]
            meta = get_meta_for_title(t)
            img = try_fetch_image(meta['image'])
            if img is not None:
                c.image(img, use_column_width=True)
            else:
                c.image(meta['image'] if isinstance(meta['image'], str) else PLACEHOLDER, use_column_width=True)
            c.markdown(f"**{t}**")
            c.caption(meta['author'])
    else:
        st.warning("Could not display Trending books (final_rating data incomplete).")

st.markdown("---")

st.subheader(f"Recommendations based on: {selected}")
recs = []
if model is not None:
    recs = recommend_with_knn(selected, model, book_pivot, k=rec_k)
elif sim_matrix is not None:
    st.warning("Similarity matrix is not used in this version of the app.")
else:
    st.warning("Neither KNN model nor similarity matrix available for recommendations.")


if not recs:
    st.info("No recommendations available for this title.")
else:
    cols = st.columns(rec_k)
    for i, t in enumerate(recs):
        c = cols[i]
        meta = get_meta_for_title(t)
        img = try_fetch_image(meta['image'])
        if img is not None:
            c.image(img, use_column_width=True)
        else:
            c.image(meta['image'] if isinstance(meta['image'], str) else PLACEHOLDER, use_column_width=True)
        c.markdown(f"**{t}**")
        c.caption(meta['author'])

st.markdown("---")
st.caption("Built from your Book Recommendation project — BookFlix")

# -------------------- Run notes --------------------
# To run the app locally:
# 1. Place your BX CSV files in ./data/ (BX-Books.csv and BX-Book-Ratings.csv)
# 2. pip install -r requirements.txt  (streamlit, scikit-learn, pandas, pillow, requests)
# 3. streamlit run app.py
