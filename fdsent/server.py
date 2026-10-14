"""A tiny local web server (Python standard library only, no extra install).

  GET  /              the dashboard page
  GET  /api/status    progress of a running job (re-run analysis / download reviews)
  POST /api/predict   classify one or many reviews (trained model + VADER)
  POST /api/rerun     re-read the reviews, retrain, rebuild the dashboard
  POST /api/collect   download fresh reviews from the Play Store, then re-run

It listens on 127.0.0.1 only, so nobody else on the network can reach it.
"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import OUT
from . import charts, models
from .clean import clean
from .dashboard import build_data, render_html, write_dashboard
from .pipeline import run_all

MAX_BODY = 400_000


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.html, self.model = "", None
        self.job = {"running": False, "kind": None, "stage": "", "pct": 0, "error": None, "done": False}


STATE = State()


def rebuild(progress=None):
    """Run the whole analysis and swap in the new page and model."""
    result = run_all(save=True, progress=progress)
    charts.save_all(result)
    write_dashboard(result, OUT / "dashboard.html")                      # the saved copy (no live buttons)
    page = render_html(build_data(result, live=True))
    with STATE.lock:
        STATE.html, STATE.model = page, result["model"]


def start_job(kind, count=1000):
    with STATE.lock:
        if STATE.job["running"]:
            return False
        STATE.job = {"running": True, "kind": kind, "stage": "Starting", "pct": 0, "error": None, "done": False}

    def update(stage, pct, lo=0, hi=100):
        with STATE.lock:
            STATE.job.update(stage=stage, pct=int(lo + (hi - lo) * pct / 100))

    def work():
        try:
            lo = 0
            if kind == "collect":
                try:
                    from .data import collect
                    import google_play_scraper  # noqa: F401
                except ImportError:
                    raise RuntimeError("The downloader is not installed. Run this once in a terminal:  pip install google-play-scraper")
                added = collect(count, progress=lambda s, p: update(s, p, 0, 40))
                if not added:
                    raise RuntimeError("No new reviews were downloaded. Check your internet connection and try again.")
                lo = 40
            rebuild(progress=lambda s, p: update(s, p, lo, 100))
            with STATE.lock:
                STATE.job.update(running=False, done=True, stage="Done", pct=100)
        except Exception as e:                   # show the reason in the page instead of crashing
            with STATE.lock:
                STATE.job.update(running=False, error=str(e) or type(e).__name__)
    threading.Thread(target=work, daemon=True).start()
    return True


def predict(texts):
    with STATE.lock:
        model = STATE.model
    texts = [str(t)[:1000] for t in texts][:300]
    if not texts:
        return []
    probs = model.predict_proba([clean(t) for t in texts])
    out = []
    for t, row in zip(texts, probs):
        chances = {c: round(float(p), 4) for c, p in zip(model.classes_, row)}
        score = models.vader_compound(t)
        out.append({"text": t, "label": max(chances, key=chances.get), "probs": chances,
                    "vader": "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral",
                    "compound": round(score, 3)})
    return out


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, separators=(",", ":")), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with STATE.lock:
                    page = STATE.html
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                with STATE.lock:
                    self._json(200, dict(STATE.job))
            elif self.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if size > MAX_BODY:
                    raise ValueError("request too large")
                req = json.loads(self.rfile.read(size) or b"{}")
                if self.path == "/api/predict":
                    texts = req.get("texts") or ([req["text"]] if "text" in req else [])
                    return self._json(200, {"results": predict(texts)})
                if self.path in ("/api/rerun", "/api/collect"):
                    count = max(100, min(5000, int(req.get("count", 1000))))
                    started = start_job("collect" if self.path == "/api/collect" else "rerun", count)
                    return self._json(200 if started else 409, {"started": started} if started else {"error": "a job is already running"})
                self._json(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
    return Handler


def serve(port=8000, open_browser=True):
    server = None
    for p in range(port, port + 30):             # if the port is busy, try the next one
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), make_handler())
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"Could not open a port between {port} and {port + 29}. Close other programs and try again.")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"\nDashboard is running at  {url}\nPress Ctrl+C in this window to stop it.")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
