import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
PASSWORD = "blue"
CATEGORIES_FILE = "categories.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # fast + accurate local model
MAX_PARAGRAPHS = 8

# =========================
# PASSWORD GATE
# =========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Protected Tool")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Unlock"):
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# =========================
# LOAD MODEL (cached)
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# =========================
# LOAD CATEGORIES
# =========================
@st.cache_data
def load_categories():
    df = pd.read_csv(CATEGORIES_FILE)
    df["label"] = df["main_category"] + " → " + df["sub_category"].fillna("")
    df["embed_text"] = (
        df["main_category"] + ". " +
        df["sub_category"].fillna("") + ". " +
        df["definition_text"]
    )
    embeddings = model.encode(df["embed_text"].tolist(), normalize_embeddings=True)
    return df, embeddings

categories_df, category_embeddings = load_categories()

# =========================
# URL CONTENT EXTRACTION (AFTER H1)
# =========================
def extract_article_after_h1(url, max_paragraphs=8):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    h1 = soup.find("h1")
    if not h1:
        return ""

    paragraphs = []
    for el in h1.find_all_next():
        if el.name == "p":
            txt = el.get_text(strip=True)
            if len(txt) > 60:
                paragraphs.append(txt)
        if len(paragraphs) >= max_paragraphs:
            break

    return " ".join(paragraphs)

# =========================
# SIMILARITY MATCHING
# =========================
def suggest_category(text):
    query_vec = model.encode([text], normalize_embeddings=True)
    scores = cosine_similarity(query_vec, category_embeddings)[0]

    categories_df["score"] = scores
    ranked = categories_df.sort_values("score", ascending=False)

    best = ranked.iloc[0]
    return best, ranked.head(5), text

# =========================
# UI
# =========================
st.title("Bluehost Article → Category Mapper")
st.caption("Semantic category mapping using **local vector embeddings** (Main Category + Sub-Category).")

mode = st.radio("Choose input mode:", ["URL → Category", "Content → Category"])

threshold = st.slider(
    "Auto-assign threshold",
    min_value=0.40,
    max_value=0.95,
    value=0.55,
    step=0.01
)

article_text = ""

# =========================
# INPUT
# =========================
if mode == "URL → Category":
    url = st.text_input("Enter article URL")
    if st.button("Suggest Category") and url:
        try:
            article_text = extract_article_after_h1(url, MAX_PARAGRAPHS)
            if len(article_text.split()) < 80:
                st.error("Could not extract meaningful article content.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to fetch URL: {e}")
            st.stop()

else:
    article_text = st.text_area("Paste article content here", height=250)
    if st.button("Suggest Category") and not article_text.strip():
        st.warning("Please paste article content.")
        st.stop()

# =========================
# PROCESS
# =========================
if article_text:
    best, top5, used_text = suggest_category(article_text)

    status = "AUTO-ASSIGN ✅" if best["score"] >= threshold else "REVIEW ⚠️"

    st.subheader("Result")
    st.write(f"**Main Category:** {best['main_category']}")
    st.write(f"**Sub-Category:** {best['sub_category']}")
    st.write(f"**Score:** `{best['score']:.3f}`")
    st.write(f"**Status:** {status}")

    st.subheader("Top suggestions")
    for _, row in top5.iterrows():
        st.write(f"- {row['label']} — {row['score']:.3f}")

    with st.expander("Text used for vector embedding"):
        st.text_area(
            "Embedding input (read-only)",
            used_text,
            height=200,
            disabled=True
        )
