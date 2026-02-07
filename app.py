import streamlit as st
import pandas as pd
import numpy as np
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
    "Semantic category mapping using local vector embeddings "
    "(Main Category + Sub-Category)."
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
# INPUT
# ---------------------------
st.subheader("Article content")

article_text = st.text_area(
    "Paste article content",
    height=250,
    placeholder="Paste H1 + introduction or full article text..."
)

submit = st.button("Submit for category mapping")

input_source = "Manually pasted content"

# ---------------------------
# HELPERS
# ---------------------------
def get_category_scores(text):
    text_embedding = model.encode(
        [text],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        text_embedding,
        category_embeddings
    )[0]

    df = categories_df.copy()
    df["score"] = scores
    return df.sort_values("score", ascending=False)


# ---------------------------
# PROCESS
# ---------------------------
if submit:
    if not article_text.strip():
        st.warning("Please paste article content before submitting.")
        st.stop()

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
            height=200,
            disabled=True
        )
