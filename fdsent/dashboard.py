"""Build output/dashboard.html: ONE self-contained web page (no server, no internet, no Streamlit).

Python does the analysis; this file packs the results (plus the trained model, so the 'Try it' box works)
into the HTML page, which draws everything with plain JavaScript.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from config import LABELS
from . import clean as clean_module
from . import models
from .analyze import NO_TOPIC, TOPICS

TEMPLATE = Path(__file__).parent / "dashboard_template.html"


def _rows(frame, **rename):
    return [{rename.get(k, k): v for k, v in r.items()} for r in frame.to_dict("records")]


def build_data(r, live=False):
    df = r["df"]
    brands = sorted(df.brand.unique())
    b_idx, l_idx = {b: i for i, b in enumerate(brands)}, {l: i for i, l in enumerate(LABELS)}
    reviews = []
    for row, topics in zip(df.itertuples(), r["topic_flags"]):
        text = row.text if len(row.text) <= 600 else row.text[:600] + "..."
        reviews.append({"b": b_idx[row.brand], "d": row.date.strftime("%Y-%m-%d %H:%M") if pd.notna(row.date) else "",
                        "r": int(row.rating), "t": text, "p": l_idx[row.pred], "v": l_idx[row.vader], "y": l_idx[row.label],
                        "pr": [round(float(row.p_negative), 2), round(float(row.p_neutral), 2), round(float(row.p_positive), 2)],
                        "tp": topics})
    dates = df.date.dropna()
    return {
        "live": live,
        "meta": {"n": len(df), "date_min": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
                 "date_max": dates.max().strftime("%Y-%m-%d") if len(dates) else "",
                 "days": int(dates.dt.date.nunique()) if len(dates) else 0, "built": date.today().isoformat()},
        "brands": brands, "labels": LABELS, "topic_names": list(TOPICS), "no_topic": NO_TOPIC,
        "reviews": reviews,
        "summary": _rows(r["summary"]), "star_dist": r["star_dist"],
        "topics": _rows(r["topics"]), "words": _rows(r["words"]),
        "trend_day": _rows(r["trend_day"], when="w", reviews="n", negative_pct="neg"),
        "trend_hour": _rows(r["trend_hour"], when="w", reviews="n", negative_pct="neg"),
        "models": _rows(r["results"]), "matrices": r["matrices"], "folds": r["folds"],
        "learning": r["learning"], "c_sweep": r["c_sweep"], "features": r["features"], "hard": r["hard"],
        "emoji": {"pos": "".join(sorted(clean_module.POSITIVE_EMOJI)), "neg": "".join(sorted(clean_module.NEGATIVE_EMOJI))},
        "jsmodel": models.export_for_js(r["model"]),
    }


def render_html(data):
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
    blob = blob.replace("</", "<\\/")          # never let the data end the <script> tag early
    return TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", blob, 1)


def write_dashboard(r, path):
    """A saved copy that opens by double-click (the live buttons are switched off in it)."""
    Path(path).write_text(render_html(build_data(r, live=False)), encoding="utf-8")
