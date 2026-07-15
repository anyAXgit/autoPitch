#!/usr/bin/env python3
"""MAHP local GUI server — ties the whole workflow into one browser app:
set project root, calibrate net ROIs on real frames, detect goal candidates,
scrub the actual footage and fine-tune cut points, then render.

Pure stdlib (http.server) so it runs offline with no extra deps beyond ffmpeg +
the project's own packages. Serves the SPA (gui/app.html), a small JSON API, and
range-enabled media so the browser <video> can scrub the big source files.

Run:  ./.venv/bin/python gui/server.py [--root .] [--port 8756]
"""
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)

STATE = {"root": PROJ}


def within_root(path):
    """Resolve a user-supplied path and confirm it stays inside the project root."""
    root = os.path.realpath(STATE["root"])
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("path escapes project root")
    return full


def games_list():
    import run_sessions
    root = STATE["root"]
    c1 = os.path.join(root, "data", "raw", "cam1")
    c2 = os.path.join(root, "data", "raw", "cam2")
    if not (os.path.isdir(c1) and os.path.isdir(c2)):
        return []
    out = []
    for g in run_sessions.pair_games(c1, c2):
        out.append({
            "game": g["game"],
            "cam1": os.path.relpath(g["cam1"], root),
            "cam2": os.path.relpath(g["cam2"], root),
            "dur1": g["dur1"], "dur2": g["dur2"], "overlap": g["overlap"],
        })
    return out


def _plan_cache_path(game):
    return os.path.join(STATE["root"], "data", "_gui", f"plan_game{game}.json")


def cached_plan(game):
    """Return the cached plan if it exists and is newer than config.yaml and
    net_rois.json (recalibrating or retuning must invalidate it)."""
    p = _plan_cache_path(game)
    if not os.path.exists(p):
        return None
    deps = [os.path.join(STATE["root"], "config.yaml"),
            os.path.join(STATE["root"], "net_rois.json")]
    newest_dep = max((os.path.getmtime(d) for d in deps if os.path.exists(d)), default=0)
    if os.path.getmtime(p) < newest_dep:
        return None
    with open(p) as f:
        return json.load(f)


def compute_plan(game):
    """Audio-only detection + planning for one game (no render). Returns plan dict."""
    from src.config import load_config
    from src.preprocess import preprocess_all
    from src.sync_engine import compute_offsets
    from src.peak_detector import detect_peaks
    from src.segment_planner import build_plan
    root = STATE["root"]
    cfg = load_config(os.path.join(root, "config.yaml"))
    gl = games_list()
    g = next(x for x in gl if x["game"] == game)
    stage = os.path.join(root, "data", "_gui", f"game{game}")
    os.makedirs(stage, exist_ok=True)
    for rel in (g["cam1"], g["cam2"]):
        src = os.path.join(root, rel)
        link = os.path.join(stage, os.path.basename(src))
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src), link)
    pre = preprocess_all(stage, os.path.join(root, "data", "_gui", "tv"),
                         os.path.join(root, "data", "_gui", f"ta{game}"),
                         cfg.fps, cfg.output_width, cfg.output_height)
    # point plan sources at the real originals (so /media can serve them for scrubbing)
    cam1_stem = os.path.splitext(os.path.basename(g["cam1"]))[0]
    pre["source"] = {c: os.path.join(root, g["cam1"] if c == cam1_stem else g["cam2"])
                     for c in pre["cams"]}
    camA = pre["cams"][0]
    offsets = compute_offsets(pre["audio"], camA) if pre["is_multicam"] else {camA: 0.0}
    peaks = detect_peaks(pre["audio"][camA], cfg)
    plan = build_plan(pre, offsets, peaks, camA, cfg)
    # rewrite src to root-relative so the browser can request /media/<rel>
    for clip in plan["clips"]:
        for seg in clip["segments"]:
            seg["src_rel"] = os.path.relpath(seg["src"], root)
    plan["game"] = game
    plan["offsets"] = offsets
    with open(_plan_cache_path(game), "w") as f:
        json.dump(plan, f)
    return plan


JOBS = {}   # job_id -> {"status": running|done|error, "progress": [i, n], ...}


def _render_job(job_id, plan, out_rel, bgm, bgm_volume):
    from src.video_editor import render_plan
    root = STATE["root"]
    job = JOBS[job_id]
    try:
        out_dir = within_root(out_rel)
        bgm_path = within_root(bgm) if bgm else None

        def on_clip(i, n, path):
            job["progress"] = [i, n]

        clips = render_plan(plan, out_dir, bgm_path, bgm_volume, on_clip=on_clip)
        job.update(status="done",
                   clips=[os.path.relpath(c, root) for c in clips])
    except Exception as e:  # noqa
        job.update(status="error", error=f"{type(e).__name__}: {e}")


def start_render(plan, out_rel, bgm=None, bgm_volume=0.15):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "progress": [0, len(plan.get("clips", []))]}
    t = threading.Thread(target=_render_job,
                         args=(job_id, plan, out_rel, bgm, bgm_volume), daemon=True)
    t.start()
    return job_id


def grab_frame(rel, t):
    full = within_root(rel)
    tmp = os.path.join(tempfile.gettempdir(), f"mahp_frame_{abs(hash((rel, t)))}.jpg")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", full,
                    "-frames:v", "1", "-vf", "scale=-2:720", tmp], check=True)
    return tmp


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._json({"error": str(msg)}, code)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _send_file(self, path, ctype=None, download=False):
        if not os.path.isfile(path):
            return self._err("not found", 404)
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = int(b) if b else size - 1
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            start, length = 0, size
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    # ---- routing ----
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send_file(os.path.join(HERE, "app.html"), "text/html")
            if u.path == "/api/state":
                rois_p = os.path.join(STATE["root"], "net_rois.json")
                return self._json({
                    "root": STATE["root"],
                    "games": games_list(),
                    "rois_exists": os.path.exists(rois_p),
                })
            if u.path == "/api/rois":
                p = os.path.join(STATE["root"], "net_rois.json")
                return self._json(json.load(open(p)) if os.path.exists(p) else {})
            if u.path == "/api/frame":
                t = float(q.get("t", ["0"])[0])
                return self._send_file(grab_frame(q["path"][0], t), "image/jpeg")
            if u.path == "/api/plan":
                game = int(q["game"][0])
                fresh = q.get("fresh", ["0"])[0] == "1"
                plan = None if fresh else cached_plan(game)
                if plan is None:
                    plan = compute_plan(game)
                else:
                    plan["cached"] = True
                return self._json(plan)
            if u.path == "/api/render_status":
                job = JOBS.get(q.get("job", [""])[0])
                return self._json(job if job else {"status": "error", "error": "unknown job"})
            if u.path.startswith("/media/"):
                return self._send_file(within_root(urllib.parse.unquote(u.path[len("/media/"):])))
            return self._err("not found", 404)
        except Exception as e:  # noqa
            return self._err(f"{type(e).__name__}: {e}", 500)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            body = self._read_json()
            if u.path == "/api/root":
                p = os.path.realpath(os.path.expanduser(body["path"]))
                if not os.path.isdir(p):
                    return self._err("not a directory")
                STATE["root"] = p
                return self._json({"root": p, "games": games_list(),
                                   "rois_exists": os.path.exists(os.path.join(p, "net_rois.json"))})
            if u.path == "/api/rois":
                json.dump(body["rois"], open(os.path.join(STATE["root"], "net_rois.json"), "w"), indent=2)
                return self._json({"ok": True})
            if u.path == "/api/render":
                job = start_render(body["plan"], body.get("out", "data/output/gui"),
                                   body.get("bgm"), body.get("bgm_volume", 0.15))
                return self._json({"job": job})
            return self._err("not found", 404)
        except Exception as e:  # noqa
            return self._err(f"{type(e).__name__}: {e}", 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=PROJ)
    ap.add_argument("--port", type=int, default=8756)
    a = ap.parse_args()
    STATE["root"] = os.path.realpath(os.path.expanduser(a.root))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"MAHP GUI on http://127.0.0.1:{a.port}  (root={STATE['root']})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
