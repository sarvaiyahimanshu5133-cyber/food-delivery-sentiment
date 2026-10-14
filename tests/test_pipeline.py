"""Run:  python -m unittest discover tests -v"""
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from fdsent import analyze, models                  # noqa: E402
from fdsent.clean import clean, rating_to_label     # noqa: E402
from fdsent.data import load_reviews                # noqa: E402


class Cleaning(unittest.TestCase):
    def test_removes_links_and_repeated_letters(self):
        self.assertEqual(clean("Soooo LATE!!! http://x.co/abc"), "soo late")

    def test_emoji_become_words(self):
        self.assertIn("negemoji", clean("worst 😡"))
        self.assertIn("posemoji", clean("great 😍"))

    def test_hindi_words_are_not_broken(self):
        self.assertEqual(clean("डिलीवरी नहीं आई"), "डिलीवरी नहीं आई")

    def test_star_labels(self):
        self.assertEqual([rating_to_label(s) for s in (1, 2, 3, 4, 5)],
                         ["negative", "negative", "neutral", "positive", "positive"])


class BadData(unittest.TestCase):
    """People will try their own CSV files, so wrong input must give a clear message."""

    def csv(self, rows):
        import tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        f.write("brand,text,rating\n" + "\n".join(rows))
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_wrong_brand_names_are_explained(self):
        path = self.csv([f"Dominos,great pizza number {i},5" for i in range(60)])
        with self.assertRaises(ValueError) as cm:
            load_reviews(path)
        self.assertIn("BRANDS", str(cm.exception))
        self.assertIn("Dominos", str(cm.exception))        # tells you what the file really contains

    def test_too_few_reviews_or_a_missing_group_is_explained(self):
        rows = [f"Swiggy,good food number {i},5" for i in range(30)] + [f"Swiggy,bad food number {i},1" for i in range(30)]
        with self.assertRaises(ValueError) as cm:
            load_reviews(self.csv(rows))                   # no 3-star reviews at all
        self.assertIn("neutral", str(cm.exception))

    def test_a_csv_without_dates_or_ids_works(self):
        rows = [f"Swiggy,review text number {i},{1 + i % 5}" for i in range(100)]
        df = load_reviews(self.csv(rows))
        self.assertEqual(len(df), 100)
        self.assertTrue(df.date.isna().all())


class Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_reviews()

    def test_loaded(self):
        self.assertGreater(len(self.df), 1000)
        self.assertEqual(set(self.df.brand), {"Swiggy", "Zomato"})
        self.assertTrue((self.df.clean.str.len() >= 2).all())
        self.assertTrue(set(self.df.label) <= {"negative", "neutral", "positive"})


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_reviews()
        cls.ev = models.evaluate(cls.df)
        cls.preds, cls.rows = cls.ev["preds"], cls.ev["rows"]

    def test_all_four_methods_are_compared(self):
        self.assertEqual({r["model"] for r in self.rows}, {"VADER", "Naive Bayes", "Linear SVM", "Logistic Regression"})

    def test_trained_models_beat_vader_on_clear_reviews(self):
        by = {r["model"]: r for r in self.rows}
        for name in ("Naive Bayes", "Linear SVM", "Logistic Regression"):
            self.assertGreater(by[name]["accuracy_pos_vs_neg"], by["VADER"]["accuracy_pos_vs_neg"], name)
        self.assertGreater(by["Logistic Regression"]["accuracy_pos_vs_neg"], 0.85)

    def test_every_review_gets_a_prediction_and_probabilities(self):
        for p in self.preds.values():
            self.assertEqual(len(p), len(self.df))
        self.assertEqual(self.ev["proba"].shape, (len(self.df), 3))
        self.assertTrue(abs(self.ev["proba"].sum(axis=1) - 1).max() < 1e-6)

    def test_learning_curve_and_features(self):
        self.assertEqual(len(models.learning_curve(self.df)), 5)
        model = models.new_model().fit(self.df.clean, self.df.label)
        top = models.top_features(model)
        self.assertIn("worst", [w for w, _ in top["negative"]])

    def test_predict_one(self):
        model = models.new_model().fit(self.df.clean, self.df.label)
        label, chances = models.predict_one(model, "Worst service, no refund, very late delivery")
        self.assertEqual(label, "negative")
        self.assertAlmostEqual(sum(chances.values()), 1.0, places=5)

    def test_model_survives_save_and_load(self):
        models.train_final(self.df)
        label, _ = models.predict_one(models.load_model(), "Amazing quick delivery, loved it")
        self.assertEqual(label, "positive")


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_reviews()
        df["pred"] = models.evaluate(df)["preds"]["Logistic Regression"]
        cls.df = df

    def test_shares_add_up_to_100(self):
        s = analyze.brand_summary(self.df)
        self.assertTrue(((s["negative_%"] + s["neutral_%"] + s["positive_%"]).round() == 100).all())

    def test_topics_and_words(self):
        t = analyze.complaint_topics(self.df)
        self.assertTrue(t["negative_reviews_%"].between(0, 100).all())
        self.assertFalse(analyze.top_words(self.df).empty)

    def test_trend(self):
        for by in ("day", "hour"):
            self.assertFalse(analyze.trend(self.df, by).empty)


class Dashboard(unittest.TestCase):
    """The HTML dashboard is built from run_all(); make sure the page is complete and self-contained."""

    @classmethod
    def setUpClass(cls):
        from fdsent.pipeline import run_all
        cls.result = run_all()

    def test_page_is_written_with_data_and_model(self):
        import json
        import tempfile
        from fdsent.dashboard import build_data, write_dashboard
        data = build_data(self.result)
        for key in ("reviews", "summary", "models", "jsmodel", "learning", "features", "trend_hour", "emoji"):
            self.assertIn(key, data)
        self.assertEqual(len(data["reviews"]), len(self.result["df"]))
        self.assertEqual(len(data["jsmodel"]["coef"]), 3)
        json.dumps(data)                                    # must be plain JSON
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dashboard.html")
            write_dashboard(self.result, path)
            with open(path, encoding="utf-8") as fh:
                html = fh.read()
        self.assertNotIn("/*DATA*/null", html)
        self.assertNotIn("https://", html.split("<script>")[0])      # no external stylesheets or fonts
        self.assertIn("Food Delivery", html)

    def test_browser_cleaning_matches_python_cleaning(self):
        """Needs Node.js. Skipped if it is not installed."""
        import shutil
        import subprocess
        import tempfile
        from fdsent.clean import clean
        from fdsent.dashboard import TEMPLATE
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        samples = ["Soooo LATE!!! http://x.co @swiggy", "worst \U0001F621\U0001F621", "\u0916\u093e\u0928\u093e \u092c\u0947\u0915\u093e\u0930 \u0925\u093e", "great \U0001F60D\U0001F44D"]
        src = TEMPLATE.read_text(encoding="utf-8")
        start, end = src.index("function cleanText"), src.index("function predictText")
        emoji = {"pos": "\U0001F60D\U0001F60A\U0001F44D", "neg": "\U0001F621\U0001F44E"}
        import json
        js = (f"const POSE=new Set([...{json.dumps(emoji['pos'])}]),NEGE=new Set([...{json.dumps(emoji['neg'])}]);" + src[start:end] +
              f"console.log(JSON.stringify({json.dumps(samples)}.map(cleanText)));")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(js)
        out = subprocess.run([node, f.name], capture_output=True, text=True, encoding="utf-8").stdout
        os.unlink(f.name)
        import fdsent.clean as c
        saved = c.POSITIVE_EMOJI, c.NEGATIVE_EMOJI
        c.POSITIVE_EMOJI, c.NEGATIVE_EMOJI = set(emoji["pos"]), set(emoji["neg"])      # same small sets as the JS side
        try:
            self.assertEqual(json.loads(out), [clean(t) for t in samples])
        finally:
            c.POSITIVE_EMOJI, c.NEGATIVE_EMOJI = saved


class LiveServer(unittest.TestCase):
    """The buttons in the dashboard talk to this server, so test it over real HTTP."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer
        from fdsent import server
        from fdsent.dashboard import build_data, render_html
        from fdsent.pipeline import run_all
        result = run_all()
        server.STATE.html = render_html(build_data(result, live=True))
        server.STATE.model = result["model"]
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler())
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        import json
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_page_is_served_in_live_mode(self):
        import urllib.request
        html = urllib.request.urlopen(self.base + "/").read().decode("utf-8")
        self.assertIn('"live":true', html)
        self.assertIn("Control panel", html)

    def test_predict_returns_model_and_vader_answers(self):
        code, out = self.post("/api/predict", {"texts": ["Worst app, no refund, very late", "Loved it, super fast delivery"]})
        self.assertEqual(code, 200)
        self.assertEqual([r["label"] for r in out["results"]], ["negative", "positive"])
        self.assertAlmostEqual(sum(out["results"][0]["probs"].values()), 1.0, places=2)
        self.assertIn(out["results"][0]["vader"], ("negative", "neutral", "positive"))

    def test_bad_requests_get_a_clear_error_not_a_crash(self):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.base + "/api/predict", b"{not json", {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(self.post("/api/nothing", {})[0], 404)

    def test_status_endpoint(self):
        import json
        import urllib.request
        status = json.loads(urllib.request.urlopen(self.base + "/api/status").read())
        self.assertIn("running", status)


if __name__ == "__main__":
    unittest.main()
