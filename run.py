#!/usr/bin/env python3
"""Food Delivery Brand Sentiment Analysis.

  python run.py            START THE DASHBOARD (analyses the reviews, opens your browser;
                           everything else is done with the buttons on the page)

Optional extras (you do not need them for the demo):
  python run.py --no-open                   do not open the browser by itself
  python run.py --port 8080                 use another port
  python run.py --save-only                 only write output/dashboard.html (works by double-click), no server
  python run.py predict "late delivery"     classify one review in this window
  python run.py check                       check that every needed package is installed
"""
import argparse
import importlib
import pathlib
import sys
import warnings
import webbrowser

if sys.version_info < (3, 9):
    sys.exit("This project needs Python 3.9 or newer.")
warnings.filterwarnings("ignore")

NEEDED = {"pandas": "pandas", "sklearn": "scikit-learn", "vaderSentiment": "vaderSentiment",
          "matplotlib": "matplotlib", "joblib": "joblib"}


def missing_packages():
    out = []
    for module, pip_name in NEEDED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            out.append(pip_name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("check", help="check the installation")
    p = sub.add_parser("predict", help="classify one review")
    p.add_argument("text")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser by itself")
    ap.add_argument("--save-only", action="store_true", help="write output/dashboard.html and stop (no server)")
    a = ap.parse_args()

    missing = missing_packages()
    if missing:
        sys.exit("Missing packages: " + ", ".join(missing) + "\nInstall everything with:\n    pip install -r requirements.txt")
    if a.cmd == "check":
        print("Everything is installed. Start the dashboard with:  python run.py")
        return

    if a.cmd == "predict":
        from fdsent import models
        try:
            model = models.load_model()
        except FileNotFoundError:
            sys.exit("No trained model yet. Start the dashboard once with:  python run.py")
        label, chances = models.predict_one(model, a.text)
        print(f"Trained model: {label}   " + "  ".join(f"{k} {v:.0%}" for k, v in chances.items()))
        print(f"VADER:         {models.vader_predict([a.text])[0]}")
        return

    from config import OUT
    from fdsent import server
    print("Food Delivery Sentiment: analysing the reviews (about 10 seconds)...")
    try:
        server.rebuild(progress=lambda stage, pct: print(f"  [{pct:3d}%] {stage}"))
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"\nThere is a problem with the reviews data:\n  {e}")
    saved = pathlib.Path(OUT) / "dashboard.html"
    print(f"(A saved copy that opens by double-click, without the live buttons: {saved.resolve()})")
    if a.save_only:
        if not a.no_open:
            webbrowser.open(saved.resolve().as_uri())
        return
    server.serve(a.port, open_browser=not a.no_open)


if __name__ == "__main__":
    main()
