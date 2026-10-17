# OracleMatch: movie recommender (web app + Python ML)

A movie recommendation project in one repo: a **live web app** that recommends from **12,081 films** (Hollywood, Bollywood and regional Indian cinema, horror), and the **Python code** behind it: a content-based model, collaborative filtering and a hybrid ranker, evaluated offline.

**Live demo:** https://av7170393-ctrl.github.io/oraclematch/ (works after you enable GitHub Pages, see "Deploy")

> This product uses the TMDB API but is not endorsed or certified by TMDB.

## Screenshots
Posters below are the app's generated fallback art (no internet when the screenshots were taken). With a free TMDB key saved in the app, real posters load.

| Home | Ranked results | Details and "Why this?" |
|---|---|---|
| ![home](docs/screenshots/01-home.png) | ![results](docs/screenshots/02-results-ranked.png) | ![details](docs/screenshots/03-details-why-this.png) |

| Filters (Hindi) | Search by description |
|---|---|
| ![filters](docs/screenshots/04-filters-hindi.png) | ![search](docs/screenshots/05-search-by-description-horror.png) |

## What the web app does
- Search a film, or pick 2 to 5 favourites ("My taste"), and get 10 recommendations as ranked cards
- **"Why this?"**: explains each result (same director, shared cast, shared themes, same franchise, same genres, language, era)
- **Search by description**: "dark sci-fi with time travel", "movies with Tom Hanks", "tamil action movies", "haunted house ghost" (keyword-based scoring, not semantic embeddings)
- Filters (genre, language, rating, year), mood picker (including "Scary"), Surprise me
- My watchlist, "More like my watchlist", "Seen it" and "Not interested" (both remove films from recommendations), optional "More variety" re-ranking, Share list
- Light and dark mode
- With a saved TMDB key: real posters, trailer link, cast, director and where to watch in India
- Everything runs in the browser from one static `index.html` (no server). Watchlist, seen, hidden films and the TMDB key are stored in your own browser only.

## Data
| Source | Films used | How they are matched |
|---|---|---|
| TMDB 5000 (movies + credits) | 4,800 | TF-IDF over plot, genres, keywords, cast, director, then re-ranked by a weighted rating |
| Indian movies (IMDb) | 4,282 (Hindi 1,888, Tamil 699, Telugu 615, Malayalam 610, Bengali 282, Kannada 122, Marathi 66) | genre, language and era; the dataset has no plots or cast |
| Horror movies (TMDB) | 2,999 added (3,484 with 50+ votes, 485 were already present) | plot wording, genres, franchise and language |

Data cleaning notes:
- Indian data: only films with a rating, year, genre and at least 300 votes. In the minor-language groups the language label was often wrong (for example Bajrangi Bhaijaan tagged Odia, Hollywood films tagged Punjabi or Nepali), so only the seven languages above are kept. Some label mistakes may remain.
- The one-line summary on Indian films is generated from the data (language, genres, year, rating, votes). It is not a real plot.
- The horror dataset ends in 2022 and has no cast or director.

## The recommender
1. Each film becomes a text document and is vectorised with TF-IDF (unigrams + bigrams).
2. Similarity is cosine similarity; the 15 closest films per film are precomputed and shipped inside `index.html`.
3. For TMDB 5000: `score = 0.85 * similarity + 0.15 * normalised weighted rating`, so results are relevant and also well rated.
4. Several liked films are combined by summing their neighbour scores.

### Offline evaluation, content model (TMDB 5000, 800 random query films, K=10)
TMDB has no user ratings, so these are proxy metrics: a recommendation counts as relevant if it shares at least half of the query film's genres.

| Model | Genre P@10 | Avg rating of results | Catalogue coverage |
|---|---|---|---|
| Plot text only | 0.253 | 6.03 | 0.706 |
| TF-IDF on all tags | **0.726** | 6.00 | 0.707 |
| + quality re-rank (0.85) (used in the app) | 0.709 | 6.74 | 0.589 |
| + quality re-rank (0.70) | 0.468 | 7.91 | 0.213 |

Adding genres, keywords, cast and director nearly triples relevance versus plot text alone. Re-ranking by rating lifts the average rating but reduces variety, so 0.85 is used.

### Collaborative filtering and hybrid (SYNTHETIC ratings)
No real ratings were available, so `src/make_synthetic_ratings.py` simulates 2,000 users with hidden tastes (140,440 ratings, see `data/DATASET_CARD.md`). **These numbers show the pipeline works; they are not real-world results.**

Rating prediction RMSE: global mean 1.036, user and item bias 0.820, ALS matrix factorisation 0.796.
Top-10 ranking on 600 held-out users (a test rating of 4 or more is relevant; the hybrid weight was tuned on separate users):

| Model | P@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.0177 | 0.0258 | 0.0256 |
| Content profile (TF-IDF) | 0.0265 | 0.0417 | 0.0380 |
| ALS matrix factorisation | 0.0232 | 0.0345 | 0.0344 |
| Item-based kNN | 0.0672 | 0.1074 | 0.1045 |
| Hybrid: ALS + content | 0.0688 | 0.1090 | 0.1041 |
| **Hybrid: item-kNN + content** | **0.0735** | **0.1165** | **0.1149** |

A bug found along the way: item-based kNN first scored P@10 = 0.001 because films with very few neighbours got inflated scores; adding shrinkage fixed it (0.067).

## Limitations
- The web app is content-based. The collaborative and hybrid models run offline on synthetic ratings only and are not in the app.
- The Python/Streamlit app (`app.py`) covers TMDB 5000 only, not the Indian and horror data.
- Description search is keyword-based, not semantic.
- Indian films have no plot, cast or director data, so their recommendations are shallower.
- Real posters need internet and, for most films, a TMDB key; without them the app shows generated posters.

## Repo layout
```
index.html                web app (deploy this file)
app.py                    Streamlit demo (content-based, TMDB 5000)
src/recommender.py        preprocessing + TF-IDF content model
src/evaluate.py           content-model evaluation
src/make_synthetic_ratings.py, src/collab.py, src/evaluate_cf.py   synthetic ratings, kNN/ALS/hybrid, evaluation
web-build/                build_data.py (raw CSVs -> data.json), build_web.py (template + data -> index.html), template.html
data/                     synthetic ratings, dataset card, instructions for the raw CSVs
results/                  evaluation outputs (json)
docs/screenshots/         screenshots used above
```

## Run it
```bash
pip install -r requirements.txt

# 1) Just use the web app: open index.html in Chrome, or host it (below)

# 2) Python experiments (needs the raw CSVs in data/, see data/README.md)
python src/evaluate.py                 # content-model table
python src/make_synthetic_ratings.py   # regenerate the synthetic ratings
python src/evaluate_cf.py              # kNN / ALS / hybrid table (about 20 seconds)
streamlit run app.py                   # Streamlit demo

# 3) Rebuild the web app from the raw CSVs
python web-build/build_web.py          # writes index.html
```

## Deploy (GitHub Pages)
Repo Settings, then Pages, choose branch `main` and folder `/ (root)`. The link appears after a minute or two.

## TMDB poster key (optional)
Create a free account at themoviedb.org, request an API key (v3), then open the app, scroll to the bottom and use "Set TMDB poster key". The key is saved only in your browser. Never commit a key to this repo.

## Tech
Python (pandas, NumPy, SciPy, scikit-learn), Streamlit, vanilla JavaScript, HTML and CSS.

## Credits
Data from TMDB (movies, credits, horror films) and IMDb (Indian films), via Kaggle datasets. Poster images and details come from TMDB when a key is saved. JustWatch provides the "where to watch" data through TMDB.
