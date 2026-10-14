# Food Delivery Brand Sentiment Analysis

> **Deep explanation of every file, the full flow and all definitions:** open [`../ARCHITECTURE.html`](../ARCHITECTURE.html) (easy to navigate; the same text is in `../ARCHITECTURE.md`).

A project for **Social Media Analytics**. It reads real Google Play reviews of **Swiggy** and **Zomato**,
finds out whether each review is happy or angry, compares the brands, shows *why* people complain,
and compares four sentiment methods with proper testing.

## Run it (one command)

**First time only:** install the packages: `pip install -r requirements.txt`
(on Windows you can skip this: double-clicking `start.bat` installs whatever is missing).

**Every time:** `python run.py`  (or double-click `start.bat`; on Mac/Linux `sh start.sh`)

That is the only command you need. It analyses the reviews (about 10 seconds), starts the dashboard and opens it in your
browser. **Everything else is done with the buttons on the page.** Press Ctrl+C in the window to stop it.

- If the browser does not open by itself, go to **http://127.0.0.1:8000** (the window prints the exact address).
- The dashboard runs only on your own computer (127.0.0.1). Nothing is uploaded anywhere.
- `output/dashboard.html` is a saved copy that opens by double-click. It shows everything except the live buttons.
- Not needed for the demo: `python run.py check` (tells you if a package is missing), `python run.py predict "late delivery"`,
  `python run.py --no-open`, `python run.py --port 8080`, `python -m unittest discover tests -v` (20 automatic checks).

## The live buttons

| Where | Button | What it does |
|---|---|---|
| **Control panel** (top) | Re-run analysis | Re-reads the reviews, retrains and compares all models again, with a progress bar. The page reloads with the new numbers. |
| | Download fresh reviews | Fetches the newest reviews from the Play Store (needs internet and `pip install google-play-scraper`), then re-runs everything. If something is missing it tells you what, in plain words. |
| **Try it** | Type a review | The trained model answers instantly, and the server also asks VADER so you see both answers. |
| | Classify all | Paste many reviews, one per line, and classify them together. |
| **Trend / Why people complain / Reviews** | sliders, drop-downs, search | Filter the charts and the review list live. |

Every card on every tab has a short grey explanation under its heading ("what this does"), and every tab has a one-line summary under the tab bar.

## What is in the dashboard

| Tab | What it shows |
|---|---|
| **Overview** | Negative / positive share and average stars per brand, star-rating histograms, plain-language findings |
| **Trend** | % negative by hour of the day or by day, with a slider to hide groups that have too few reviews |
| **Why people complain** | Topic bars (delivery, refund, app problems...), common words, and a viewer that lists real complaints by topic |
| **Model Lab** | Leaderboard of 4 methods, confusion matrices, learning curve, tuning curve, the words the model learned, and real reviews where VADER fails |
| **Try it** | Type any review: probabilities, which words pushed the answer, and the strongest clues |
| **Reviews** | Search, filter and sort all 1,400 reviews, with the model's and VADER's answer next to the star rating |
| **Method** | The pipeline step by step, key ideas, and an honest limitations list |

## Use your own reviews (your own scenario)

Full guide with a tested example: section 12.2 of `ARCHITECTURE.html`. The short version:

1. **No files:** open the *Try it* tab and type a review, or paste many into **Many reviews at once**.
2. **Your own CSV:** save it as `data/reviews.csv` with the columns `brand`, `text`, `rating` (1 to 5); `date` is optional. Change `BRANDS` in `config.py` so the names match your `brand` column **exactly**. Then run `python run.py`. You need at least 50 reviews and at least 5 in each group (1-2 stars, 3 stars, 4-5 stars); otherwise the program stops with a clear message.
3. **Your own topics:** add a line to `TOPICS` in `fdsent/analyze.py`, for example `"Hygiene": r"dirty|hair|hygiene|insect"`. It appears everywhere automatically.
4. **Another Play Store app:** add its package id (the part after `id=` in the app's Play Store address) to `BRANDS`, then press **Download fresh reviews**.

## The idea in simple words

Thousands of people write reviews. Nobody can read them all, so we make the computer read them:

1. **Collect** real reviews (a star rating plus text).
2. **Clean** the text: lowercase, remove links, shorten `soooo` to `soo`, turn 😡 into the word `negemoji`.
3. **Teach models** what a negative review sounds like. The answer key is the star rating (1-2 = negative, 3 = neutral, 4-5 = positive).
4. **Test them honestly** with 5-fold cross-validation, and compare them with VADER, a ready-made English word list.
5. **Analyse** what share of each brand's reviews is negative, and what the negative reviews talk about.
6. **Show it** in the dashboard.

**Real-life example:** it is like a shop owner with 1,400 feedback slips. Instead of reading all of them, the owner
wants to know how many customers are unhappy, and whether the problem is delivery, food, or refunds.

## How it works (explain it like this)

- **TF-IDF** turns text into numbers. A rare, telling word (*refund*) gets more weight than a common one (*the*). We use single words *and* word pairs, so "not good" is different from "good".
- **Four methods:** VADER (no training), Naive Bayes (counts words per class), Linear SVM (best dividing line) and Logistic Regression (gives probabilities, weights can be read). We keep Logistic Regression.
- **Cross-validation (5-fold):** split the reviews into 5 groups, train on 4, test on the 5th, repeat 5 times. Every review is predicted by a model that never saw it, like testing a student on questions they have not memorised.
- **Accuracy vs F1:** about 67% of reviews are positive, so a model that always says "positive" is already 67% accurate. F1 is measured per class so a rare class cannot hide.
- **How the dashboard works:** Python does the analysis and packs the results and the trained model into one HTML page. A small built-in server (`fdsent/server.py`) serves that page and answers the live buttons. JavaScript draws the charts and also runs the same TF-IDF + Logistic Regression maths, so the page still works if you open the saved copy. `tests/` checks that the browser gives the same answer as Python.

## Results on the shipped data (run `python run.py` to regenerate)

| | VADER | Naive Bayes | Linear SVM | Logistic Regression |
|---|---|---|---|---|
| Accuracy on clearly happy/unhappy reviews | 76% | 91% | 90% | **91%** |
| Accuracy including neutral | 74% | 87% | 86% | **87%** |

The three trained models are within 1 point of each other, which is less than the spread between folds, so we do not claim one is better. All three clearly beat VADER.

| | Swiggy | Zomato |
|---|---|---|
| Negative reviews | 32% | 23% |
| Average stars | 3.56 | 3.82 |
| Top complaint topics | Delivery/delay 53%, Refund/support 45%, App problems 32% | Delivery/delay 47%, Refund/support 41%, App problems 35% |

## Demo script (3 minutes, all clicks, no typing commands)

1. **Overview:** "Swiggy has more negative reviews than Zomato, 32% against 23%."
2. **Why people complain:** "Delivery delay and refund/support are the top two reasons for both." Then pick *Refund / support* in the viewer and read real complaints.
3. **Model Lab:** "VADER gets 76% right, our trained models get about 91%." Open the *Where VADER fails* list. Then the learning curve: "more data would help a little."
4. **Try it:** the default review ("Delivery was very late and the food was cold") is **negative**, VADER says **neutral**. Click the other example buttons. Then paste a few reviews into *Many reviews at once*.
5. **Control panel:** click **Re-run analysis** and show the progress bar. "The whole pipeline runs live."
6. **Method:** read the limitations before they ask.

## Files (about 800 lines of Python plus the HTML page)

```
run.py                       command line (analyse / predict / collect / check)
config.py                    brands and file locations
data/reviews.csv             1,400 real Play Store reviews (Swiggy + Zomato)
fdsent/clean.py              text cleaning + stars -> label
fdsent/data.py               load the CSV; download new reviews
fdsent/models.py             the 4 methods, cross-validation, learning curve, tuning  <- the important file
fdsent/analyze.py            brand summary, trend, complaint topics, common words
fdsent/pipeline.py           runs everything in order
fdsent/dashboard.py          packs the results into the HTML page
fdsent/server.py             the small local server behind the live buttons (Python standard library only)
start.bat / start.sh         double-click launchers
fdsent/dashboard_template.html   the dashboard (HTML + CSS + JavaScript, no libraries)
fdsent/charts.py             saves PNG charts (handy for slides)
tests/test_pipeline.py       automatic tests
output/                      dashboard.html, charts and CSV tables made by `python run.py`
```

## Questions you may be asked

**Why use star ratings as the answer key?** Labelling 1,400 reviews by hand takes days, and stars are free. They are imperfect: a 5-star review can still contain a complaint, and the *Where both are wrong* list in Model Lab shows real examples. So accuracy means "agrees with the star rating".

**Why is neutral so bad?** Only 4% of the reviews are 3 stars, and a 3-star review mixes good and bad, so the words give no clear signal. We tried forcing neutral predictions (`class_weight="balanced"`): overall accuracy fell from 87% to 67% and neutral did not improve. So we kept the better model and say clearly that neutral is weak.

**Why did you not remove stop words like "not"?** "not good" and "good" mean opposite things. We remove stop words only when listing the most common words.

**Why is VADER bad here?** It is built for general English social-media text. It does not know that *refund*, *cancelled* or *late* are bad news in a delivery review, so it often answers "neutral".

**Why not Streamlit?** It needs a live websocket connection, which some computers and browsers block (it can stay stuck on "Connecting"). A plain local web server with normal requests is more reliable, and it needs no extra install.

**Why not a deep-learning model like BERT?** With 1,400 short reviews, a simple model is easier to explain, trains in seconds and already reaches about 91%. A larger model would be the next step with much more data.

## Limitations (say these before they ask)

- **Only 3 days of reviews** (the Play Store returns the newest first), so there is no long-term trend. `python run.py collect --count 5000` on a machine with internet gets more, and the Trend tab fills in by itself.
- Play Store reviewers are mostly very happy or very angry, so this is not a random sample of customers.
- **Hinglish** (Hindi in English letters) is only 1-2% of the reviews, so the model is weak on it.
- Complaint topics come from keyword lists, so unusual wording is missed.
- Brand percentages use cross-validated predictions. The "Try it" box uses the model trained on all reviews.
