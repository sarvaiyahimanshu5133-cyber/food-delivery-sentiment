"""Turn a raw review into clean text, and turn a star rating into a sentiment label."""
import re
import unicodedata

# A few emojis carry a lot of sentiment, so we turn them into words instead of throwing them away.
POSITIVE_EMOJI = set("😍😊😀😃😄😁👍👌❤💖💕🔥🥰😋😎🙏👏🤩💯✅⭐")
NEGATIVE_EMOJI = set("😡🤬👎😤😠😞😢😭🤮😒😑🙄💔❌😖😣😫")


def clean(text):
    """'Soooo LATE!!! 😡 http://x.co' -> 'soo late negemoji'"""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", t)                        # links and @mentions
    t = "".join(" posemoji " if c in POSITIVE_EMOJI else " negemoji " if c in NEGATIVE_EMOJI else c for c in t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                                     # soooo -> soo
    t = re.sub(r"[^\w\s'\u0900-\u097f]", " ", t)                             # drop punctuation, keep Hindi letters
    return re.sub(r"\s+", " ", t).strip()


def rating_to_label(stars):
    """1-2 stars = negative, 3 = neutral, 4-5 = positive."""
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"
