"""Offline evaluation of the recommender variants.

TMDB has no user ratings, so we use PROXY metrics (be honest about this in the README):
  - Genre Precision@K : a recommendation is 'relevant' if its genre Jaccard overlap with the query movie >= 0.5
  - Avg rating        : mean vote_average of recommended movies (quality)
  - Coverage          : share of the catalogue that ever appears in a top-K list (diversity / anti-popularity-bias)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from recommender import MovieRecommender, load_movies

K = 10
N_QUERIES = 800
SEED = 42


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def evaluate(df, get_topk, name, queries):
    precs, ratings, seen = [], [], set()
    for q in queries:
        top = get_topk(q)
        qg = df.at[q, "genres_list"]
        rel = [jaccard(qg, df.at[i, "genres_list"]) >= 0.5 for i in top]
        precs.append(np.mean(rel))
        ratings.append(df.loc[top, "vote_average"].mean())
        seen.update(top)
    return {
        "model": name,
        f"genre_precision@{K}": round(float(np.mean(precs)), 3),
        "avg_rating_of_recs": round(float(np.mean(ratings)), 2),
        "catalog_coverage": round(len(seen) / len(df), 3),
    }


def main():
    df = load_movies()
    rng = np.random.default_rng(SEED)
    queries = rng.choice(len(df), size=N_QUERIES, replace=False)
    results = []

    # 1) Baseline: overview text only
    tf = TfidfVectorizer(stop_words="english", max_features=20000)
    M = tf.fit_transform(df["overview"].str.lower())

    def overview_only(i):
        s = linear_kernel(M[i], M).ravel()
        s[i] = -1
        return np.argsort(-s)[:K]

    results.append(evaluate(df, overview_only, "Baseline: overview only", queries))

    # 2) Full tags (overview + genres + keywords + cast + director), no re-rank
    rec = MovieRecommender(df, alpha=1.0)

    def tags_only(i):
        s = linear_kernel(rec.matrix[i], rec.matrix).ravel()
        s[i] = -1
        return np.argsort(-s)[:K]

    results.append(evaluate(df, tags_only, "Content TF-IDF (all tags)", queries))

    # 3) Full tags + quality re-ranking at different alphas
    for alpha in (0.85, 0.7):
        r = MovieRecommender(df, alpha=alpha)

        def rerank(i, r=r):
            s = linear_kernel(r.matrix[i], r.matrix).ravel()
            sc = r.alpha * s + (1 - r.alpha) * r.quality
            sc[i] = -1
            return np.argsort(-sc)[:K]

        results.append(evaluate(df, rerank, f"Content + quality re-rank (alpha={alpha})", queries))

    out = Path(__file__).resolve().parent.parent / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"{'model':45s} P@{K}   rating  coverage")
    for r in results:
        print(f"{r['model']:45s} {r[f'genre_precision@{K}']:.3f}  {r['avg_rating_of_recs']:.2f}    {r['catalog_coverage']:.3f}")


if __name__ == "__main__":
    main()
