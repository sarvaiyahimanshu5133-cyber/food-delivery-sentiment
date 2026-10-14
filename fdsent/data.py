"""Load the reviews from CSV, and (optionally) download fresh ones from the Google Play Store."""
import time

import pandas as pd

from config import BRANDS, DATA, LABELS
from .clean import clean, rating_to_label


def load_reviews(path=DATA, brands=None):
    """Read the CSV and add two columns: `clean` (cleaned text) and `label` (from the stars)."""
    df = pd.read_csv(path)
    missing = {"brand", "text", "rating"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} must have the columns brand, text, rating (missing: {sorted(missing)})")
    if "review_id" not in df:
        df["review_id"] = range(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.NaT
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["text", "rating"])
    found = sorted(df.brand.dropna().astype(str).unique())[:8]
    df = df[df.rating.between(1, 5) & df.brand.isin(brands or BRANDS)].drop_duplicates("review_id").copy()
    if df.empty:
        raise ValueError(f"No reviews left after filtering. The brand names in config.py (BRANDS: {', '.join(brands or BRANDS)}) "
                         f"must match the 'brand' column of {path} exactly. That column contains: {', '.join(found) or 'nothing'}.")
    df["clean"] = df["text"].map(clean)
    df = df[df["clean"].str.len() >= 2].copy()
    df["label"] = df["rating"].astype(int).map(rating_to_label)
    counts = df.label.value_counts()
    too_few = [f"{label}: {int(counts.get(label, 0))}" for label in LABELS if counts.get(label, 0) < 5]
    if len(df) < 50 or too_few:
        raise ValueError(f"Not enough reviews to train a model (found {len(df)}). Aim for at least 50 in total and at least 5 in each group: "
                         f"1-2 stars = negative, 3 stars = neutral, 4-5 stars = positive. Too few: {', '.join(too_few) or 'none'}.")
    return df.reset_index(drop=True)


def collect(count=2000, pause=1.0, progress=None):
    """Download the newest reviews of every brand in config.BRANDS (needs internet and
    `pip install google-play-scraper`) and merge them into data/reviews.csv. Returns how many NEW reviews arrived."""
    from google_play_scraper import Sort, reviews

    rows = []
    for n, (brand, app_id) in enumerate(BRANDS.items()):
        got, token = 0, None
        while got < count:
            try:
                batch, token = reviews(app_id, lang="en", country="in", sort=Sort.NEWEST,
                                       count=min(200, count - got), continuation_token=token)
            except Exception as e:                      # network problem or rate limit: keep what we have
                print(f"  {brand}: stopped early ({type(e).__name__}: {e})")
                break
            if not batch:
                break
            rows += [{"review_id": r["reviewId"], "brand": brand, "date": r["at"], "rating": r["score"],
                      "text": r["content"], "source": "play_store"} for r in batch]
            got += len(batch)
            print(f"  {brand}: {got} reviews", end="\r")
            if progress:
                progress(f"Downloading {brand} reviews ({got} of {count})", int(100 * (n + got / count) / len(BRANDS)))
            if getattr(token, "token", 1) is None:      # no more pages
                break
            time.sleep(pause)
        print(f"  {brand}: {got} reviews downloaded")
    if not rows:
        return 0
    old = pd.read_csv(DATA) if DATA.exists() else pd.DataFrame()
    merged = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("review_id")
    merged.to_csv(DATA, index=False)
    added = len(merged) - len(old)
    print(f"Saved {len(merged)} reviews to {DATA} ({added} new)")
    return added
