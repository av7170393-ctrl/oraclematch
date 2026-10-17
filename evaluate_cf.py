"""Evaluate popularity, item-kNN, ALS, content profile and hybrid on the (synthetic) ratings."""
import json
import time
from pathlib import Path

import numpy as np

from collab import (BiasedALS, ContentProfile, ItemKNN, PopularityModel, hybrid, load_ratings, split, to_csr)
from recommender import DATA_DIR, MovieRecommender, load_movies

K = 10
REL = 4.0          # a test rating >= 4 counts as "relevant"


def rank_metrics(S, train_csr, test_by_user, users):
    S = S.copy(); S[train_csr[users].nonzero()] = -1e9
    top = np.argsort(-S, axis=1)[:, :K]
    P, R, N, seen = [], [], [], set()
    disc = 1 / np.log2(np.arange(2, K + 2))
    for row, u in zip(top, users):
        rel = test_by_user.get(u, set())
        if not rel: continue
        hit = np.array([i in rel for i in row], float)
        P.append(hit.mean()); R.append(hit.sum() / len(rel))
        N.append((hit * disc).sum() / disc[:min(len(rel), K)].sum()); seen.update(row.tolist())
    return dict(precision=round(float(np.mean(P)), 4), recall=round(float(np.mean(R)), 4), ndcg=round(float(np.mean(N)), 4)), seen


def main():
    t0 = time.time()
    df = load_movies(); rec = MovieRecommender(df)
    ratings = load_ratings(Path(DATA_DIR) / "ratings_synthetic.csv", df)
    nu, ni = ratings.u.nunique(), len(df)
    train, test = split(ratings)
    print(f"ratings {len(ratings):,} | train {len(train):,} | test {len(test):,} | users {nu}")
    train_csr = to_csr(train, nu, ni)
    test_by_user = {}
    for u, i, r in zip(test.u, test.i, test.rating):
        if r >= REL: test_by_user.setdefault(u, set()).add(i)
    users_all = np.array(sorted(test_by_user))
    rng = np.random.default_rng(3); rng.shuffle(users_all)
    tune_u, eval_u = users_all[:300], users_all[300:900]            # tune hybrid weight on one set, report on another

    # ---- rating prediction (RMSE on held-out ratings) ----
    als = BiasedALS().fit(train, nu, ni)
    mu = train.rating.mean()
    bi = np.bincount(train.i, weights=train.rating - mu, minlength=ni) / (np.bincount(train.i, minlength=ni) + 10)
    bu = np.bincount(train.u, weights=train.rating - mu - bi[train.i], minlength=nu) / (np.bincount(train.u, minlength=nu) + 10)
    rmse = lambda p: float(np.sqrt(np.mean((p - test.rating.to_numpy()) ** 2)))
    out = {"rmse": {
        "Global mean": round(rmse(np.full(len(test), mu)), 4),
        "User + item bias": round(rmse(np.clip(mu + bu[test.u] + bi[test.i], .5, 5)), 4),
        "ALS matrix factorisation": round(rmse(als.predict_pairs(test.u.to_numpy(), test.i.to_numpy())), 4)}}
    print("RMSE:", out["rmse"])

    # ---- top-10 ranking ----
    pop = PopularityModel().fit(train, nu, ni)
    knn = ItemKNN().fit(train, nu, ni)
    con = ContentProfile(rec.matrix).fit(train, nu, ni)
    mask_t = (train_csr[tune_u] != 0).toarray(); mask_e = (train_csr[eval_u] != 0).toarray()

    def tune(cf):
        best_w, best = 0.5, -1
        for w in (0.2, 0.35, 0.5, 0.65, 0.8):
            m, _ = rank_metrics(hybrid(cf.scores(tune_u), con.scores(tune_u), mask_t, w), train_csr, test_by_user, tune_u)
            if m["ndcg"] > best: best, best_w = m["ndcg"], w
        return best_w
    w_als, w_knn = tune(als), tune(knn)
    print("hybrid CF weights chosen on tuning users: ALS", w_als, "| kNN", w_knn)

    models = {
        "Popularity": pop.scores(eval_u),
        "Item-based kNN": knn.scores(eval_u),
        "ALS matrix factorisation": als.scores(eval_u),
        "Content profile (TF-IDF)": con.scores(eval_u),
        f"Hybrid: ALS + content (w={w_als})": hybrid(als.scores(eval_u), con.scores(eval_u), mask_e, w_als),
        f"Hybrid: item-kNN + content (w={w_knn})": hybrid(knn.scores(eval_u), con.scores(eval_u), mask_e, w_knn),
    }
    best_w = {"als": w_als, "knn": w_knn}
    out["ranking"] = {}
    for name, S in models.items():
        m, seen = rank_metrics(S, train_csr, test_by_user, eval_u)
        m["coverage"] = round(len(seen) / ni, 3)
        out["ranking"][name] = m
        print(f"{name:38s} P@{K}={m['precision']:.4f}  R@{K}={m['recall']:.4f}  NDCG@{K}={m['ndcg']:.4f}  coverage={m['coverage']:.3f}")
    out["meta"] = {"ratings": len(ratings), "users": int(nu), "eval_users": len(eval_u), "k": K, "relevant_rating": REL, "hybrid_cf_weight": best_w,
                   "note": "SYNTHETIC ratings; metrics demonstrate the pipeline, not real-world performance."}
    res = Path(__file__).resolve().parent.parent / "results"; res.mkdir(exist_ok=True)
    (res / "results_cf.json").write_text(json.dumps(out, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
