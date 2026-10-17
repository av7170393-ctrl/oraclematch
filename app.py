import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from recommender import MovieRecommender, load_movies  # noqa: E402

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading movies and building model...")
def get_model():
    return MovieRecommender(load_movies(), alpha=0.85)


@st.cache_data(show_spinner=False)
def get_poster(title):
    """Optional posters via OMDb. Set OMDB_API_KEY in env or .streamlit/secrets.toml."""
    key = os.environ.get("OMDB_API_KEY") or st.secrets.get("OMDB_API_KEY", None)
    if not key:
        return None
    try:
        data = requests.get("http://www.omdbapi.com/", params={"apikey": key, "t": title}, timeout=5).json()
        poster = data.get("Poster")
        return poster if data.get("Response") == "True" and poster != "N/A" else None
    except Exception:
        return None


def show_cards(results):
    for row_start in range(0, len(results), 5):
        cols = st.columns(5)
        for col, (_, m) in zip(cols, results.iloc[row_start:row_start + 5].iterrows()):
            with col:
                poster = get_poster(m["title"])
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"**{m['title']}**")
                year = f"{int(m['year'])} · " if m["year"] == m["year"] else ""
                st.caption(f"{year}⭐ {m['vote_average']}")
                st.caption(", ".join(m["genres_list"][:3]))
                if "overview" in m:
                    with st.expander("Plot"):
                        st.write(m["overview"])


model = get_model()
titles = sorted(model.df["title"].tolist())

st.title("🎬 Movie Recommender")
st.caption("Content-based (TF-IDF on plot, genres, keywords, cast, director) + quality re-ranking. Data: TMDB 5000.")

tab1, tab2 = st.tabs(["Similar to one movie", "Based on movies I like"])

with tab1:
    movie = st.selectbox("Pick a movie", titles, index=None, placeholder="Type to search...")
    n = st.slider("How many recommendations", 5, 15, 10, step=5, key="n1")
    boost = st.toggle("Prefer highly rated movies", value=True)
    if movie:
        show_cards(model.similar(movie, n=n, rerank=boost))

with tab2:
    liked = st.multiselect("Pick 2-5 movies you like", titles, max_selections=5)
    n2 = st.slider("How many recommendations", 5, 15, 10, step=5, key="n2")
    if len(liked) >= 2:
        show_cards(model.recommend_from_liked(liked, n=n2))
    else:
        st.info("Select at least 2 movies.")
