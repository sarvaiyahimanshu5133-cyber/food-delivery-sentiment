"""The whole analysis in one function, used by run.py and by the dashboard's buttons."""
import json

import pandas as pd

from config import OUT
from . import analyze, models
from .data import load_reviews


def run_all(save=False, progress=None):
    """progress(stage_text, percent) is called along the way so the dashboard can show a progress bar."""
    step = progress or (lambda stage, pct: None)
    step("Reading and cleaning the reviews", 5)
    df = load_reviews()
    step("Comparing 4 methods with 5-fold cross-validation", 15)
    ev = models.evaluate(df)
    df["pred"] = ev["preds"]["Logistic Regression"]       # brand analysis uses the trained model's (cross-validated) predictions
    df["vader"] = ev["preds"]["VADER"]
    for i, label in enumerate(models.LABELS):
        df[f"p_{label}"] = ev["proba"][:, i]
    step("Training the final model", 45)
    model = models.train_final(df)
    step("Analysing brands, complaint topics and trends", 55)
    result = {
        "df": df, "model": model, "results": pd.DataFrame(ev["rows"]), "matrices": ev["matrices"], "folds": ev["folds"],
        "summary": analyze.brand_summary(df), "topics": analyze.complaint_topics(df), "words": analyze.top_words(df, 12),
        "star_dist": analyze.star_distribution(df), "topic_flags": analyze.topic_flags(df),
        "trend_day": analyze.trend(df, "day", min_reviews=1), "trend_hour": analyze.trend(df, "hour", min_reviews=1),
    }
    step("Learning curve: does more data help?", 70)
    result["learning"] = models.learning_curve(df)
    step("Tuning the model setting C", 85)
    result["c_sweep"] = models.c_sweep(df)
    step("Collecting what the model learned", 95)
    result["features"] = models.top_features(model)
    result["hard"] = models.hard_examples(df)
    if save:
        OUT.mkdir(exist_ok=True)
        result["results"].to_csv(OUT / "model_results.csv", index=False)
        result["summary"].to_csv(OUT / "brand_summary.csv", index=False)
        result["topics"].to_csv(OUT / "complaint_topics.csv", index=False)
        result["words"].to_csv(OUT / "top_words.csv", index=False)
        df[["review_id", "brand", "date", "rating", "text", "label", "pred", "vader"]].to_csv(OUT / "predictions.csv", index=False)
        (OUT / "confusion.json").write_text(json.dumps(result["matrices"]), encoding="utf-8")
    step("Done", 100)
    return result
