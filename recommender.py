"""Movie recommender core: preprocessing + content-based model + quality re-ranking.

Data: TMDB 5000 Movies + Credits (Kaggle).
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------- preprocessing ----------
def _names(text, limit=None):
    """Parse a JSON-like string column and return list of 'name' values."""
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    names = [d["name"] for d in items if "name" in d]
    return names[:limit] if limit else names


def _director(text):
    try:
        for d in ast.literal_eval(text):
            if d.get("job") == "Director":
                return [d["name"]]
    except (ValueError, SyntaxError):
        pass
    return []


def _squash(words):
    """'Sam Worthington' -> 'samworthington' so first names don't match across people."""
    return [w.replace(" ", "").lower() for w in words]


def load_movies(data_dir=DATA_DIR):
    movies = pd.read_csv(Path(data_dir) / "tmdb_5000_movies.csv")
    credits = pd.read_csv(Path(data_dir) / "tmdb_5000_credits.csv")
    df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")

    df["genres_list"] = df["genres"].apply(_names)
    df["keywords_list"] = df["keywords"].apply(_names)
    df["cast_list"] = df["cast"].apply(lambda x: _names(x, limit=3))
    df["director_list"] = df["crew"].apply(_director)
    df["overview"] = df["overview"].fillna("")
    df["year"] = pd.to_datetime(df["release_date"], format="%d-%m-%Y", errors="coerce").dt.year

    # one "tags" document per movie
    df["tags"] = (
        df["overview"].str.lower()
        + " "
        + df["genres_list"].apply(lambda g: " ".join(_squash(g) * 2))  # genres weighted x2
        + " "
        + df["keywords_list"].apply(lambda k: " ".join(_squash(k)))
        + " "
        + df["cast_list"].apply(lambda c: " ".join(_squash(c)))
        + " "
        + df["director_list"].apply(lambda d: " ".join(_squash(d) * 2))  # director x2
    )

    # IMDB weighted rating: balances vote_average with vote_count
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.60)
    v, R = df["vote_count"], df["vote_average"]
    df["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

    keep = [
        "id", "title", "year", "overview", "genres_list", "vote_average", "vote_count",
        "popularity", "weighted_rating", "tags",
    ]
    return df[keep].drop_duplicates(subset="title").reset_index(drop=True)


# ---------- model ----------
class MovieRecommender:
    def __init__(self, df, alpha=0.85):
        """alpha = weight of content similarity; (1 - alpha) = weight of movie quality."""
        self.df = df
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=2)
        self.matrix = self.vectorizer.fit_transform(df["tags"])  # sparse, no 184 MB pickle needed
        self.title_to_idx = pd.Series(df.index, index=df["title"])
        wr = df["weighted_rating"].to_numpy()
        self.quality = (wr - wr.min()) / (wr.max() - wr.min())

    def similar(self, title, n=10, rerank=True):
        """Return DataFrame of the n most similar movies to `title`."""
        if title not in self.title_to_idx:
            raise KeyError(f"'{title}' not in dataset")
        idx = self.title_to_idx[title]
        sim = linear_kernel(self.matrix[idx], self.matrix).ravel()
        score = self.alpha * sim + (1 - self.alpha) * self.quality if rerank else sim
        score[idx] = -1  # exclude the movie itself
        top = np.argsort(-score)[:n]
        out = self.df.iloc[top][["title", "year", "genres_list", "vote_average", "overview"]].copy()
        out["similarity"] = sim[top].round(3)
        out["score"] = score[top].round(3)
        return out.reset_index(drop=True)

    def recommend_from_liked(self, titles, n=10):
        """Recommend from several liked movies (average of their TF-IDF vectors)."""
        idxs = [self.title_to_idx[t] for t in titles if t in self.title_to_idx]
        if not idxs:
            raise KeyError("none of the given titles are in the dataset")
        profile = np.asarray(self.matrix[idxs].mean(axis=0))
        sim = linear_kernel(profile, self.matrix).ravel()
        score = self.alpha * sim + (1 - self.alpha) * self.quality
        score[idxs] = -1
        top = np.argsort(-score)[:n]
        return self.df.iloc[top][["title", "year", "genres_list", "vote_average"]].reset_index(drop=True)
