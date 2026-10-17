"""Collaborative filtering + hybrid recommenders.

Models: popularity baseline, item-based kNN (adjusted cosine), biased ALS matrix factorisation,
content profile (TF-IDF from the existing content model) and a hybrid blend of CF + content.
Input format = MovieLens style (userId, movieId, rating); movieId must be a TMDB id.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_ratings(path, df):
    r = pd.read_csv(path)
    id2col = {m: i for i, m in enumerate(df["id"])}
    r = r[r.movieId.isin(id2col)].copy()
    r["u"] = pd.factorize(r.userId)[0]
    r["i"] = r.movieId.map(id2col)
    return r.reset_index(drop=True)


def split(r, test_frac=0.2, seed=1):
    rng = np.random.default_rng(seed)
    test = rng.random(len(r)) < test_frac
    return r[~test].reset_index(drop=True), r[test].reset_index(drop=True)


def to_csr(r, n_users, n_items, col="rating"):
    return sp.csr_matrix((r[col].to_numpy(float), (r.u, r.i)), shape=(n_users, n_items))


class PopularityModel:
    def fit(self, train, n_users, n_items):
        cnt = np.bincount(train.i, minlength=n_items).astype(float)
        mean = np.bincount(train.i, weights=train.rating, minlength=n_items) / np.maximum(cnt, 1)
        self.s = np.log1p(cnt) * mean          # popular AND liked
        return self

    def scores(self, users):
        return np.tile(self.s, (len(users), 1))


class BiasedALS:
    def __init__(self, k=24, reg=8.0, iters=8, seed=0):
        self.k, self.reg, self.iters, self.seed = k, reg, iters, seed

    def fit(self, train, n_users, n_items):
        u, i, r = train.u.to_numpy(), train.i.to_numpy(), train.rating.to_numpy(float)
        self.mu = r.mean()
        bi = np.bincount(i, weights=r - self.mu, minlength=n_items) / (np.bincount(i, minlength=n_items) + 10)
        bu = np.bincount(u, weights=r - self.mu - bi[i], minlength=n_users) / (np.bincount(u, minlength=n_users) + 10)
        self.bi, self.bu = bi, bu
        res = r - self.mu - bu[u] - bi[i]
        R = sp.csr_matrix((res, (u, i)), shape=(n_users, n_items)); Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        P = rng.normal(0, 0.1, (n_users, self.k)); Q = rng.normal(0, 0.1, (n_items, self.k))
        eye = self.reg * np.eye(self.k)
        def solve(M, F, other):
            out = np.zeros((M.shape[0], self.k))
            for a in range(M.shape[0]):
                s, e = M.indptr[a], M.indptr[a + 1]
                if e == s: continue
                Fo = other[M.indices[s:e]]
                out[a] = np.linalg.solve(Fo.T @ Fo + eye, Fo.T @ M.data[s:e])
            return out
        for _ in range(self.iters):
            P = solve(R, P, Q); Q = solve(Rt, Q, P)
        self.P, self.Q = P, Q
        return self

    def predict_pairs(self, users, items):
        return np.clip(self.mu + self.bu[users] + self.bi[items] + (self.P[users] * self.Q[items]).sum(1), 0.5, 5)

    def scores(self, users):
        return self.mu + self.bu[users][:, None] + self.bi[None, :] + self.P[users] @ self.Q.T


class ItemKNN:
    def __init__(self, topk=40, shrink=4.0):
        self.topk, self.shrink = topk, shrink          # shrink: damps scores of films with few neighbours

    def fit(self, train, n_users, n_items):
        R = to_csr(train, n_users, n_items)
        um = np.asarray(R.sum(1)).ravel() / np.maximum(np.diff(R.indptr), 1)
        C = R.copy().tocoo(); C.data = C.data - um[C.row]
        self.C = C.tocsr()
        norms = np.sqrt(np.asarray(self.C.multiply(self.C).sum(0)).ravel()) + 1e-9
        S = (self.C.T @ self.C).toarray() / np.outer(norms, norms)
        np.fill_diagonal(S, 0)
        keep = np.argpartition(-np.abs(S), self.topk, axis=1)[:, :self.topk]       # prune to top-k neighbours per item
        mask = np.zeros_like(S, dtype=bool); np.put_along_axis(mask, keep, True, axis=1)
        self.S = sp.csr_matrix(np.where(mask, S, 0))
        self.B = (self.C != 0).astype(float)
        return self

    def scores(self, users):
        num = (self.C[users] @ self.S).toarray()
        den = (self.B[users] @ abs(self.S)).toarray() + self.shrink
        return num / den


class ContentProfile:
    """User profile = weighted average of TF-IDF vectors of films they rated above their own mean."""
    def __init__(self, tfidf):
        self.T = tfidf  # items x terms (L2-normalised)

    def fit(self, train, n_users, n_items):
        R = to_csr(train, n_users, n_items)
        um = np.asarray(R.sum(1)).ravel() / np.maximum(np.diff(R.indptr), 1)
        W = R.copy().tocoo(); W.data = np.maximum(W.data - um[W.row], 0); self.W = W.tocsr()
        self.W.eliminate_zeros()
        return self

    def scores(self, users):
        prof = self.W[users] @ self.T
        norm = np.sqrt(np.asarray(prof.multiply(prof).sum(1))).ravel() + 1e-9
        return ((prof @ self.T.T).toarray()) / norm[:, None]


def zscore(S, mask):
    S = S.copy(); S[mask] = np.nan
    return (S - np.nanmean(S, 1, keepdims=True)) / (np.nanstd(S, 1, keepdims=True) + 1e-9)


def hybrid(cf, content, mask, w):
    """w = weight of the collaborative score, (1-w) = content score (each z-scored per user)."""
    return np.nan_to_num(w * zscore(cf, mask) + (1 - w) * zscore(content, mask), nan=-1e9)
