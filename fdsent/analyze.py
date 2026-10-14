"""Turn predictions into the numbers a brand manager wants: who is liked more, when, and why people complain."""
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from config import LABELS

# Complaint topics. A review "mentions" a topic if it contains one of its keywords (English + Hinglish).
TOPICS = {
    "Delivery / delay": r"deliver\w*|late|delay\w*|rider|partner|wait\w*|time|slow|cancel\w*|der|डिलीवरी",
    "Food quality": r"food|taste\w*|cold|stale|quality|raw|smell\w*|khana|thanda|spoil\w*|hair|fresh|baasi",
    "Wrong / missing items": r"missing|wrong|incomplete|different|item\w*|not received|never received|nahi mila",
    "Price / charges": r"price\w*|expensive|costly|charge\w*|fees?|surge|mehenga|mehnga|paisa|money|overpriced|discount\w*|offer\w*|coupon\w*",
    "Refund / support": r"refund\w*|support|customer care|service|complaint\w*|response|helpline|executive|resolve\w*|cheat\w*|fraud|froud|scam|रिफंड",
    "App problems": r"app|crash\w*|bug\w*|glitch\w*|login|otp|payment\w*|upi|update\w*|error|location|hang\w*|loading",
}
_TOPIC_RX = {t: re.compile(rf"\b(?:{p})\b") for t, p in TOPICS.items()}
NO_TOPIC = "No specific topic (e.g. 'worst app')"

# Words that say nothing about WHY someone is unhappy (shown words are counted after removing these).
SKIP = set(ENGLISH_STOP_WORDS) | set("""app order ordered ordering swiggy zomato hai hain ho tha thi the ka ki ke ko se me mein
ye yeh wo woh aur bhi to toh ek kar kya hi negemoji posemoji very really just get got one even don didn doesn isn wasn won couldn wouldn""".split())


def brand_summary(df):
    rows = []
    for brand, g in df.groupby("brand"):
        share = g.pred.value_counts(normalize=True).reindex(LABELS, fill_value=0) * 100
        rows.append({"brand": brand, "reviews": len(g), "avg_stars": round(g.rating.mean(), 2),
                     **{f"{label}_%": round(share[label], 1) for label in LABELS}})
    return pd.DataFrame(rows)


def trend(df, by="day", min_reviews=10):
    """% of negative reviews per day, or per hour of the day. Buckets with too few reviews are left out."""
    d = df.dropna(subset=["date"]).copy()
    d["when"] = d.date.dt.strftime("%Y-%m-%d") if by == "day" else d.date.dt.hour
    g = d.groupby(["brand", "when"]).agg(reviews=("pred", "size"),
                                          negative_pct=("pred", lambda s: round((s == "negative").mean() * 100, 1))).reset_index()
    return g[g.reviews >= min_reviews]


def complaint_topics(df):
    """Among NEGATIVE reviews, what share mention each topic? (one review can mention several)"""
    rows = []
    for brand, g in df[df.pred == "negative"].groupby("brand"):
        hits = pd.DataFrame({t: g.clean.str.contains(rx) for t, rx in _TOPIC_RX.items()})
        for t in TOPICS:
            rows.append({"brand": brand, "topic": t, "negative_reviews_%": round(hits[t].mean() * 100, 1)})
        rows.append({"brand": brand, "topic": NO_TOPIC, "negative_reviews_%": round((~hits.any(axis=1)).mean() * 100, 1)})
    return pd.DataFrame(rows)


def top_words(df, k=10):
    """Most common words in each brand's negative reviews."""
    rows = []
    for brand, g in df[df.pred == "negative"].groupby("brand"):
        if len(g) < 5:
            continue
        cv = CountVectorizer(stop_words=sorted(SKIP), token_pattern=r"(?u)\b[^\W\d_]{3,}\b", binary=True)
        counts = cv.fit_transform(g.clean).sum(axis=0).A1
        for i in counts.argsort()[::-1][:k]:
            rows.append({"brand": brand, "word": cv.get_feature_names_out()[i], "reviews": int(counts[i]),
                         "share_%": round(counts[i] / len(g) * 100, 1)})
    return pd.DataFrame(rows)


def topic_flags(df):
    """For every review: the list of topic numbers it mentions (so the dashboard can filter by topic)."""
    names = list(TOPICS)
    return [[i for i, t in enumerate(names) if _TOPIC_RX[t].search(c)] for c in df.clean]


def star_distribution(df):
    """How many 1,2,3,4,5-star reviews each brand has."""
    return {b: [int((g.rating == s).sum()) for s in range(1, 6)] for b, g in df.groupby("brand")}
