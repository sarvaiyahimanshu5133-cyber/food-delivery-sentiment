"""Four ways to find the sentiment of a review, compared fairly.

  VADER                a ready-made English word list ("great" = +, "worst" = -). No training.
  Naive Bayes          counts how often each word appears in positive / negative reviews.
  Linear SVM           draws the best dividing line between positive and negative reviews.
  Logistic Regression  like the SVM, but it gives PROBABILITIES and its weights can be read as
                       "this word pushes towards negative". This is the model we keep and use.

All trained models learn from the same TF-IDF numbers (rare, telling words weigh more than "the"),
with the star rating as the answer key.

Fair comparison = 5-fold cross-validation: split the reviews into 5 parts, train on 4 parts, predict the 5th,
repeat. Every review is therefore predicted by a model that never saw it.
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import LABELS, MODEL
from .clean import clean

_vader = SentimentIntensityAnalyzer()


def split_words(text):
    return text.split()          # cleaning already put single spaces between words


def vader_predict(texts):
    out = []
    for t in texts:
        score = _vader.polarity_scores(str(t))["compound"]       # -1 (very negative) ... +1 (very positive)
        out.append("positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral")
    return out


def vader_compound(text):
    """VADER's raw score for one review: -1 (very negative) ... +1 (very positive)."""
    return float(_vader.polarity_scores(str(text))["compound"])


def tfidf():
    # ngram_range=(1, 2): single words AND pairs, so "not good" is not confused with "good".
    return TfidfVectorizer(tokenizer=split_words, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def new_model(C=1.0):
    # (We tried class_weight="balanced" to push the model towards "neutral"; it made accuracy much
    #  worse, because only about 4% of reviews are 3 stars. See the README.)
    return make_pipeline(tfidf(), LogisticRegression(C=C, max_iter=2000))


TRAINED = {
    "Naive Bayes": lambda: make_pipeline(tfidf(), MultinomialNB(alpha=0.3)),
    "Linear SVM": lambda: make_pipeline(tfidf(), LinearSVC(C=0.5)),
    "Logistic Regression": new_model,
}


def scores(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    clear = y_true != "neutral"          # reviews that are clearly happy (4-5 stars) or unhappy (1-2 stars)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3),
            "accuracy_pos_vs_neg": round(float((y_true[clear] == y_pred[clear]).mean()), 3),
            "macro_f1": round(float(np.mean(f1)), 3),
            **{f"f1_{label}": round(float(v), 3) for label, v in zip(LABELS, f1)}}


def _folds(seed):
    return StratifiedKFold(5, shuffle=True, random_state=seed)


def cv_predict(make, texts, y, seed=42):
    """Out-of-fold predictions (and probabilities, if the model has them) + the accuracy of each fold."""
    texts, y = np.asarray(texts), np.asarray(y)
    pred, proba, accs = np.empty(len(y), dtype=object), None, []
    for train, test in _folds(seed).split(texts, y):
        m = make().fit(texts[train], y[train])
        pred[test] = m.predict(texts[test])
        accs.append(accuracy_score(y[test], pred[test]))
        if hasattr(m, "predict_proba"):
            proba = np.zeros((len(y), len(LABELS))) if proba is None else proba
            proba[test] = m.predict_proba(texts[test])       # columns follow m.classes_ = LABELS (alphabetical)
    return list(pred), proba, accs


def evaluate(df, seed=42):
    """Predict every review with all four methods. Returns a dict with predictions, the results table,
    confusion matrices, fold accuracies and the probabilities of the Logistic Regression."""
    preds, folds, proba = {}, {}, None
    preds["VADER"] = vader_predict(df.text)
    vader_arr = np.asarray(preds["VADER"])
    folds["VADER"] = [accuracy_score(df.label.values[t], vader_arr[t]) for _, t in _folds(seed).split(df.clean, df.label)]
    for name, make in TRAINED.items():
        p, pr, accs = cv_predict(make, df.clean, df.label, seed)
        preds[name], folds[name] = p, accs
        if name == "Logistic Regression":
            proba = pr
    rows, matrices = [], {}
    for name, p in preds.items():
        rows.append({"model": name, **scores(df.label, p),
                     "fold_mean": round(float(np.mean(folds[name])), 3), "fold_std": round(float(np.std(folds[name])), 3)})
        matrices[name] = confusion_matrix(df.label, p, labels=LABELS).tolist()
    return {"preds": preds, "proba": proba, "rows": rows, "matrices": matrices,
            "folds": {k: [round(float(a), 3) for a in v] for k, v in folds.items()}}


def learning_curve(df, seed=42, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Does the model get better with more reviews? (train on a fraction of each training fold)"""
    texts, y, rng, out = df.clean.values, df.label.values, np.random.RandomState(seed), []
    for f in fractions:
        accs, sizes = [], []
        for train, test in _folds(seed).split(texts, y):
            sub = rng.choice(train, max(40, int(len(train) * f)), replace=False)
            accs.append(accuracy_score(y[test], new_model().fit(texts[sub], y[sub]).predict(texts[test])))
            sizes.append(len(sub))
        out.append({"train_size": int(np.mean(sizes)), "accuracy": round(float(np.mean(accs)), 3),
                    "std": round(float(np.std(accs)), 3)})
    return out


def c_sweep(df, seed=42, values=(0.1, 0.3, 1, 3, 10, 30)):
    """Logistic Regression has one main setting, C (bigger C = trust the training data more)."""
    out = []
    for c in values:
        pred, _, _ = cv_predict(lambda c=c: new_model(c), df.clean, df.label, seed)
        s = scores(df.label, pred)
        out.append({"C": c, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"]})
    return out


def top_features(model, k=15):
    """The words/word-pairs with the biggest weight for each class (what the model learned)."""
    names = model.named_steps["tfidfvectorizer"].get_feature_names_out()
    lr = model.named_steps["logisticregression"]
    return {c: [[str(names[j]), round(float(lr.coef_[i][j]), 3)] for j in np.argsort(lr.coef_[i])[::-1][:k]]
            for i, c in enumerate(lr.classes_)}


def hard_examples(df, k=8, seed=7):
    """Real reviews where VADER is wrong. Split into: the trained model gets them right / both are wrong."""
    # only English / Roman-script reviews of readable length, so the examples are easy to judge
    base = df[(df.label != "neutral") & (df.clean.str.split().str.len().between(5, 45))
              & df.text.map(lambda t: t.isascii())]
    cols = ["brand", "rating", "text", "label", "pred", "vader"]
    fixed = base[(base.vader != base.label) & (base.pred == base.label)]
    both = base[(base.vader != base.label) & (base.pred != base.label)]
    pick = lambda d: d.sample(min(k, len(d)), random_state=seed)[cols].to_dict("records")
    return {"model_fixes_vader": pick(fixed), "both_wrong": pick(both),
            "counts": {"model_fixes_vader": int(len(fixed)), "both_wrong": int(len(both))}}


def train_final(df):
    """Train on ALL reviews and save it. This is the model used by 'predict' and by the dashboard's Try-it box."""
    model = new_model().fit(df.clean, df.label)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    return model


def load_model():
    return joblib.load(MODEL)


def predict_one(model, text):
    """Returns (label, {label: probability}) for one review."""
    probs = model.predict_proba([clean(text)])[0]
    chances = {c: float(p) for c, p in zip(model.classes_, probs)}
    return max(chances, key=chances.get), chances


def export_for_js(model):
    """Everything the dashboard needs to run the SAME model in the browser (no server needed)."""
    vec, lr = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    vocab = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    return {"classes": list(lr.classes_), "vocab": vocab,
            "idf": [round(float(x), 5) for x in vec.idf_],
            "coef": [[round(float(x), 5) for x in row] for row in lr.coef_],
            "intercept": [round(float(x), 5) for x in lr.intercept_]}
