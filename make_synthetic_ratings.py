"""Generate a SYNTHETIC ratings dataset (simulated users) for TMDB 5000 movies.

This is NOT real user data. Each simulated user has hidden tastes (genre affinity, favourite era,
strictness); they tend to watch popular films they like, and rate them from affinity + film quality + noise.
Output schema matches MovieLens (userId, movieId, rating, timestamp), where movieId = TMDB id.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from recommender import load_movies, DATA_DIR

N_USERS, SEED = 2000, 7
OUT = Path(DATA_DIR) / "ratings_synthetic.csv"


def main():
    rng = np.random.default_rng(SEED)
    df = load_movies()
    raw = pd.read_csv(Path(DATA_DIR) / "tmdb_5000_movies.csv")[["id", "original_language"]]
    df = df.merge(raw, on="id", how="left").reset_index(drop=True)
    genres = sorted({g for l in df.genres_list for g in l})
    gi = {g: i for i, g in enumerate(genres)}
    X = np.zeros((len(df), len(genres)))
    for m, l in enumerate(df.genres_list):
        for g in l:
            X[m, gi[g]] = 1
    gX = X / np.maximum(X.sum(1, keepdims=True), 1)                       # movie genre mix
    pop = df.popularity.to_numpy() ** 0.8
    year = df.year.fillna(2000).to_numpy()
    english = (df.original_language == "en").to_numpy()
    wr = df.weighted_rating.to_numpy(); qz = (wr - wr.mean()) / wr.std()   # film quality z-score

    rows = []
    for u in range(1, N_USERS + 1):
        taste = rng.dirichlet(np.full(len(genres), 0.35)) * len(genres)    # hidden genre affinity
        aff = gX @ taste; z = (aff - aff.mean()) / (aff.std() + 1e-9)
        fav_year = np.clip(rng.normal(2005, 9), 1975, 2014)
        world = rng.random() < 0.10                                        # some users like non-English films
        w = pop * np.exp(1.1 * z) * np.exp(-((year - fav_year) / 18) ** 2 / 2)
        w = w * np.where(english, 1.0, 3.0 if world else 0.4)
        n = int(np.clip(np.exp(rng.normal(np.log(45), 0.8)) + 10, 20, 500))
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False, p=w / w.sum())
        bias = rng.normal(0, 0.4)                                          # strict vs generous rater
        r = 2.55 + 0.7 * np.clip(z[idx], -2, 2) + 0.45 * qz[idx] + bias + rng.normal(0, 0.6, len(idx))
        r = np.clip(np.round(r * 2) / 2, 0.5, 5.0)
        ts = rng.integers(1451606400, 1546214400, len(idx))                # 2016-2018
        rows.append(pd.DataFrame({"userId": u, "movieId": df.id.to_numpy()[idx], "rating": r, "timestamp": ts}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)
    dens = len(out) / (N_USERS * len(df))
    print(f"ratings: {len(out):,} | users: {N_USERS} | movies rated: {out.movieId.nunique()} | density: {dens:.2%}")
    print("mean rating:", round(out.rating.mean(), 2), "| ratings per user (median):", int(out.groupby('userId').size().median()))
    print(out.rating.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
