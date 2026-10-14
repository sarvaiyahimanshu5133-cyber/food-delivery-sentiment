"""Project settings. Change BRANDS to analyse other apps."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "reviews.csv"            # real Google Play reviews
OUT = ROOT / "output"                           # charts and tables are written here
MODEL = ROOT / "models" / "sentiment.joblib"    # the trained model

# Brand -> Google Play package id (the id= part of the app's Play Store URL)
BRANDS = {
    "Swiggy": "in.swiggy.android",
    "Zomato": "com.application.zomato",
}
LABELS = ["negative", "neutral", "positive"]
