"""Build the data behind index.html from the three raw CSV files in ../data/.

  1. TMDB 5000 movies + credits  -> content similarity (TF-IDF) with quality re-rank, plus director/cast/keywords
  2. Indian movies (IMDb)        -> genre + language + era similarity (no plots or cast in that dataset)
  3. Horror movies (TMDB)        -> plot-wording + genre + franchise + language similarity

Each film is one list: [title, year, genres, rating, summary, id, director, cast, keywords, language, poster_path, franchise]
and every film has its 15 closest films (indexes + similarity). Output: data.json
"""
import ast, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
from recommender import MovieRecommender, load_movies, _director, _names  # noqa: E402

K = 15


def clip(t, n=170):
    t = (t or "").strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def step1_tmdb5000():
    df = load_movies().sort_values("popularity", ascending=False).reset_index(drop=True)
    rec = MovieRecommender(df, alpha=0.85)
    M, nb, sc = rec.matrix, [], []
    for s in range(0, len(df), 400):
        sims = (M[s:s + 400] @ M.T).toarray()
        scores = rec.alpha * sims + (1 - rec.alpha) * rec.quality[None, :]
        for r in range(scores.shape[0]): scores[r, s + r] = -1
        idx = np.argsort(-scores, axis=1)[:, :K]
        for r in range(idx.shape[0]):
            nb.append(idx[r].tolist()); sc.append([int(round(max(sims[r, j], 0) * 1000)) for j in idx[r]])
    movies = pd.read_csv(DATA / "tmdb_5000_movies.csv"); credits = pd.read_csv(DATA / "tmdb_5000_credits.csv")
    raw = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")
    bad = re.compile(r"stinger|credits|^3d$|^sequel$|^based on|^duringcredits|aftercredits")
    facets = {}
    for r in raw.itertuples():
        d = _director(r.crew)
        facets[int(r.id)] = (d[0] if d else "", _names(r.cast, 3), [k.lower() for k in _names(r.keywords)[:10] if not bad.search(k.lower())], r.original_language)
    recs = []
    for r in df.itertuples():
        f = facets.get(int(r.id), ("", [], [], "en"))
        recs.append([r.title, int(r.year) if r.year == r.year else 0, r.genres_list[:3], round(float(r.vote_average), 1), clip(r.overview), int(r.id), f[0], f[1], f[2], f[3]])
    return {"m": recs, "nb": nb, "sc": sc}


def step2_indian(base):
    raw = pd.read_csv(DATA / "indian_movies.csv")
    raw.columns = ["imdb", "title", "year", "runtime", "rating", "votes", "genre", "lang"]
    raw["y"] = pd.to_numeric(raw.year.astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    raw["r"] = pd.to_numeric(raw.rating, errors="coerce")
    raw["v"] = pd.to_numeric(raw.votes.astype(str).str.replace(",", "", regex=False), errors="coerce")
    d = raw.drop_duplicates("imdb").dropna(subset=["r", "v", "y"]); d = d[d.v >= 300].copy()
    GMAP, DROP = {"Sci-Fi": "Science Fiction", "Musical": "Music"}, {"-", "Short", "Talk-Show", "Reality-TV", "News", "Game-Show", ""}
    def genres(s):
        out = []
        for g in str(s).split(","):
            g = GMAP.get(g.strip(), g.strip())
            if g and g not in DROP and g not in out: out.append(g)
        return out[:3]
    d["g"] = d.genre.apply(genres); d = d[d.g.str.len() > 0]
    # Only languages whose labels look reliable; in the minor-language groups the label is often wrong
    # (e.g. Bajrangi Bhaijaan tagged Odia, Hollywood films tagged Punjabi or Nepali), so those rows are dropped.
    LC = {"hindi": "hi", "tamil": "ta", "telugu": "te", "malayalam": "ml", "kannada": "kn", "bengali": "bn", "marathi": "mr"}
    LN = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada", "bn": "Bengali", "mr": "Marathi"}
    d["lc"] = d.lang.str.strip().str.lower().map(LC); d = d.dropna(subset=["lc"]); d["title"] = d.title.str.strip()
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    seen = {}
    for m in base["m"]: seen.setdefault(norm(m[0]), []).append(m[1])
    dup = d.apply(lambda r_: any(abs(r_.y - y) <= 1 for y in seen.get(norm(r_.title), [])), axis=1)
    d = d[~dup].sort_values("v", ascending=False).reset_index(drop=True)
    C, mm = d.r.mean(), 1000
    wr = (d.v / (d.v + mm)) * d.r + (mm / (d.v + mm)) * C
    q = ((wr - wr.min()) / (wr.max() - wr.min())).to_numpy()
    lv = np.log1p(d.v); lv = ((lv - lv.min()) / (lv.max() - lv.min())).to_numpy()
    docs = [" ".join(["g_" + g.replace(" ", "_") for g in gl] * 3 + ["l_" + lc] * 2 + ["d_%d" % (int(y) // 10 * 10)]) for gl, lc, y in zip(d.g, d.lc, d.y)]
    X = TfidfVectorizer(analyzer=str.split, norm="l2").fit_transform(docs)
    off, nb, sc = len(base["m"]), [], []
    for s in range(0, len(d), 500):
        sim = (X[s:s + 500] @ X.T).toarray()
        tot = 0.75 * sim + 0.15 * q[None, :] + 0.10 * lv[None, :]
        for r_ in range(tot.shape[0]): tot[r_, s + r_] = -1
        idx = np.argsort(-tot, axis=1)[:, :K]
        for r_ in range(idx.shape[0]):
            nb.append([int(j) + off for j in idx[r_]]); sc.append([int(round(max(sim[r_, j], 0) * 1000)) for j in idx[r_]])
    for r_ in d.itertuples():
        ov = f"{LN[r_.lc]} {', '.join(x.lower() for x in r_.g)} film from {int(r_.y)}. IMDb rating {r_.r} from {int(r_.v):,} votes."
        base["m"].append([r_.title, int(r_.y), r_.g, round(float(r_.r), 1), ov, r_.imdb, "", [], [], r_.lc])
    base["nb"] += nb; base["sc"] += sc
    return base


def step3_horror(base):
    df = pd.read_csv(DATA / "horror_movies.csv")
    df["year"] = pd.to_datetime(df.release_date, errors="coerce").dt.year
    H = df[(df.status == "Released") & df.overview.notna() & df.year.notna() & (df.vote_count >= 50)].copy()
    H = H.sort_values("vote_count", ascending=False).reset_index(drop=True)
    exist = {m[5]: i for i, m in enumerate(base["m"]) if isinstance(m[5], int)}
    H["old"] = H.id.map(exist)
    for r in H.itertuples():                         # films already in the app: attach poster path and franchise
        if r.old == r.old:
            m = base["m"][int(r.old)]
            while len(m) < 12: m.append("")
            if isinstance(r.poster_path, str): m[10] = r.poster_path
            if isinstance(r.collection_name, str): m[11] = re.sub(r" Collection$", "", r.collection_name)
    def genres(s):
        g = [x.strip() for x in str(s).split(",") if x.strip() and x.strip() != "TV Movie"]
        return (["Horror"] + [x for x in g if x != "Horror"])[:3]
    H["g"] = H.genre_names.apply(genres)
    off, new_mask = len(base["m"]), H.old.isna()
    H["idx"] = H.old.astype("float"); H.loc[new_mask, "idx"] = off + np.arange(int(new_mask.sum())); H["idx"] = H.idx.astype(int)
    docs = []
    for r in H.itertuples():
        col = re.sub(r"[^a-z0-9]+", "_", str(r.collection_name).lower()) if isinstance(r.collection_name, str) else ""
        docs.append(f"{r.overview.lower()} " + " ".join(["g_" + g.lower().replace(" ", "_") for g in r.g[1:]] * 2) + (f" col_{col}" * 4 if col else "") + f" lang_{r.original_language}" * 2)
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True).fit_transform(docs)
    C, mm = H.vote_average.mean(), 300
    wr = (H.vote_count / (H.vote_count + mm)) * H.vote_average + (mm / (H.vote_count + mm)) * C
    q = ((wr - wr.min()) / (wr.max() - wr.min())).to_numpy()
    lv = np.log1p(H.vote_count); lv = ((lv - lv.min()) / (lv.max() - lv.min())).to_numpy()
    idx_arr, nb, sc, recs = H.idx.to_numpy(), [], [], []
    for s in range(0, len(H), 400):
        sim = (X[s:s + 400] @ X.T).toarray()
        tot = 0.82 * sim + 0.12 * q[None, :] + 0.06 * lv[None, :]
        for r_ in range(tot.shape[0]): tot[r_, s + r_] = -1
        top = np.argsort(-tot, axis=1)[:, :K]
        for r_ in range(top.shape[0]):
            row = s + r_
            if H.old.iloc[row] == H.old.iloc[row] and not np.isnan(H.old.iloc[row]): continue   # existing film keeps its list
            nb.append([int(idx_arr[j]) for j in top[r_]]); sc.append([int(round(max(sim[r_, j], 0) * 1000)) for j in top[r_]])
            c = H.collection_name.iloc[row]; pp = H.poster_path.iloc[row]
            recs.append([H.title.iloc[row].strip(), int(H.year.iloc[row]), H.g.iloc[row], round(float(H.vote_average.iloc[row]), 1), clip(H.overview.iloc[row]),
                         int(H.id.iloc[row]), "", [], [], H.original_language.iloc[row], pp if isinstance(pp, str) else "", re.sub(r" Collection$", "", c) if isinstance(c, str) else ""])
    base["m"] += recs; base["nb"] += nb; base["sc"] += sc
    return base


def build():
    data = step3_horror(step2_indian(step1_tmdb5000()))
    assert len(data["m"]) == len(data["nb"]) == len(data["sc"])
    return data


if __name__ == "__main__":
    data = build()
    (ROOT / "web-build" / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print("films:", len(data["m"]))
