import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# BASIC CONFIG
# ---------------------------
st.set_page_config(
    page_title="Bluehost Article → Category Mapper",
    layout="wide"
)

APP_PASSWORD = "blue"

# ---------------------------
# PASSWORD GATE
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Internal Access")
    password = st.text_input("Enter password", type="password")
    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.stop()

# ---------------------------
# LOAD MODEL (cached)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# LOAD CATEGORY DATA
# ---------------------------
@st.cache_data
def load_categories():
    df = pd.read_csv("categories.csv")
    df["definition_text"] = df["definition_text"].fillna("")
    df["full_text"] = (
        df["main_category"] + " | " +
        df["sub_category"].fillna("") + " | " +
        df["definition_text"]
    )
    return df

categories_df = load_categories()
category_embeddings = model.encode(
    categories_df["full_text"].tolist(),
    normalize_embeddings=True
)

# ---------------------------
# UI
# ---------------------------
st.title("Bluehost Article → Category Mapper")
st.caption(
    "Semantic category mapping using **local vector embeddings** "
    "(Main Category + Sub-Category)."
)

mode = st.radio(
    "Choose input mode:",
    ["URL → Category", "Content → Category"]
)

threshold = st.slider(
    "Auto-assign threshold",
    min_value=0.40,
    max_value=0.95,
    value=0.55,
    step=0.01
)

st.divider()

# ---------------------------
# HELPERS
# ---------------------------
def fetch_article_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    h1 = soup.find("h1")
    if not h1:
        return ""

    paragraphs = []
    for tag in h1.find_all_next(["p", "h2"]):
        if tag.name == "h2":
            break
        text = tag.get_text(strip=True)
        if len(text.split()) > 5:
            paragraphs.append(text)

    return " ".join(paragraphs)


def get_category_scores(text):
    text_embedding = model.encode(
        [text],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        text_embedding,
        category_embeddings
    )[0]

    categories_df["score"] = scores
    return categories_df.sort_values("score", ascending=False)


# ---------------------------
# INPUT
# ---------------------------
article_text = ""
input_source = ""

if mode == "URL → Category":
    url = st.text_input("Enter article URL")
    if st.button("Suggest Category") and url:
        try:
            article_text = fetch_article_text(url)
            input_source = "URL content after <h1>"
            if not article_text:
                st.error("Could not extract meaningful article content.")
                st.stop()
        except requests.exceptions.HTTPError as e:
            if "403" in str(e):
                st.warning(
                    "⚠️ This site blocks cloud requests.\n\n"
                    "Please use **Content → Category** mode instead."
                )
                st.stop()
            else:
                st.error(f"Failed to fetch URL: {e}")
                st.stop()

else:
    article_text = st.text_area(
        "Paste article content here",
        height=220
    )
    input_source = "Manually pasted content"

# ---------------------------
# PROCESS
# ---------------------------
if article_text:
    ranked = get_category_scores(article_text)
    top = ranked.iloc[0]

    status = "AUTO-ASSIGN ✅" if top.score >= threshold else "REVIEW ⚠️"

    st.subheader("Result")
    st.markdown(f"**Suggested Main Category:** {top.main_category}")
    st.markdown(
        f"**Suggested Sub-Category:** "
        f"{top.sub_category if pd.notna(top.sub_category) else '—'}"
    )
    st.markdown(f"**Score:** `{top.score:.3f}`")
    st.markdown(f"**Status:** {status}")

    st.subheader("Top suggestions")
    for _, row in ranked.head(5).iterrows():
        st.markdown(
            f"- **{row.main_category} → "
            f"{row.sub_category if pd.notna(row.sub_category) else '—'}** "
            f"— `{row.score:.3f}`"
        )

    with st.expander("Text used for vector embedding"):
        st.caption(input_source)
        st.text_area(
            "Embedding input (read-only)",
            article_text,
            height=200
        )
