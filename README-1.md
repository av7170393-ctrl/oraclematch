# Data

Shipped in this repo:
- `ratings_synthetic.csv`: SYNTHETIC ratings for the Python experiments (see `DATASET_CARD.md`). Not real user data.

Raw files you need only if you want to rebuild `index.html` or re-run the Python scripts (they are git-ignored because of size).
Put them in this folder with exactly these names:

| File | What it is |
|---|---|
| `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv` | Kaggle "TMDB 5000 Movie Dataset" |
| `indian_movies.csv` | Kaggle Indian movies dataset from IMDb (columns: ID, Name, Year, Timing, Rating, Votes, Genre, Language) |
| `horror_movies.csv` | Kaggle horror movies dataset from TMDB (32,540 films, up to 2022) |
