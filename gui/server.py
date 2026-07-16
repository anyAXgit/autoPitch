#!/usr/bin/env python3
"""autoPitch local GUI server — ties the whole workflow into one browser app:
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
import re
import shutil
import tempfile
import threading
import urllib.parse
import uuid
import platform
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


def output_dir_path(path):
    """Resolve render output. Relative paths stay inside the project; absolute
    paths are allowed because the user explicitly selected them via the native
    folder picker."""
    if os.path.isabs(path):
        return os.path.realpath(os.path.expanduser(path))
    return within_root(path)


def choose_native_dir(initial=""):
    initial = os.path.realpath(os.path.expanduser(initial)) if initial else STATE["root"]
    if not os.path.isdir(initial):
        initial = STATE["root"]
    if platform.system() == "Darwin":
        script = (
            'POSIX path of (choose folder with prompt "출력 폴더 선택" '
            f'default location POSIX file "{initial}")'
        )
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askdirectory(initialdir=initial, title="출력 폴더 선택")
        root.destroy()
        return picked or None
    except Exception:
        return None


VIDEO_EXTS = (".mp4", ".mov", ".m4v")


def _cam_sort_key(name):
    m = re.search(r"(\d+)$", name)
    return (0, int(m.group(1))) if m else (1, name)


def _list_videos(d):
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )


def _probe(path):
    from datetime import datetime

    def q(entries):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", entries,
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().splitlines()

    dur = float(q("format=duration")[0])
    ct = q("format_tags=creation_time")
    start = None
    if ct and ct[0]:
        start = datetime.fromisoformat(ct[0].replace("Z", "+00:00")).timestamp()
    return start, dur


def camera_dirs():
    root = STATE["root"]
    raw = os.path.join(root, "data", "raw")
    if not os.path.isdir(raw):
        return []
    dirs = []
    for name in sorted(os.listdir(raw), key=_cam_sort_key):
        full = os.path.join(raw, name)
        if os.path.isdir(full) and re.match(r"^cam\d+$", name, re.I):
            dirs.append((name.lower(), full))
    return dirs


def camera_status():
    root = STATE["root"]
    raw = os.path.join(root, "data", "raw")
    dirs = []
    for cam_id, full in camera_dirs():
        files = _list_videos(full)
        dirs.append({
            "id": cam_id,
            "path": os.path.relpath(full, root),
            "count": len(files),
            "files": [os.path.basename(f) for f in files[:4]],
        })
    return {
        "raw_exists": os.path.isdir(raw),
        "config_exists": os.path.exists(os.path.join(root, "config.yaml")),
        "cameras": dirs,
    }


def state_payload():
    root = STATE["root"]
    rois_p = os.path.join(root, "net_rois.json")
    games, setup_error = [], None
    try:
        games = games_list()
    except Exception as e:  # noqa
        setup_error = f"{type(e).__name__}: {e}"
    return {
        "root": root,
        "games": games,
        "rois_exists": os.path.exists(rois_p),
        "setup": camera_status(),
        "setup_error": setup_error,
    }


def list_dirs(rel=""):
    rel = rel.strip("/")
    full = within_root(rel)
    if not os.path.isdir(full):
        raise ValueError("not a directory")
    root = os.path.realpath(STATE["root"])
    cur = os.path.relpath(full, root)
    if cur == ".":
        cur = ""
    dirs = []
    for name in sorted(os.listdir(full), key=str.lower):
        if name.startswith("."):
            continue
        p = os.path.join(full, name)
        if os.path.isdir(p):
            dirs.append(name)
    parent = os.path.dirname(cur) if cur else None
    return {"path": cur, "parent": parent, "dirs": dirs}


def games_list():
    """Group one to four camera folders into games by capture time.

    The first camera folder (cam1) is the anchor. For each cam1 clip, the nearest
    unused clip from every other camera is attached. A single-camera shoot simply
    yields one game per cam1 clip.
    """
    root = STATE["root"]
    dirs = camera_dirs()
    if not dirs:
        return []
    tracks = {}
    for cam_id, d in dirs[:4]:
        clips = [{"path": p, "start": s, "dur": dur}
                 for p in _list_videos(d) for s, dur in [_probe(p)]]
        if not clips:
            return []
        if len(dirs) > 1 and any(c["start"] is None for c in clips):
            raise SystemExit("A clip is missing creation_time metadata; cannot pair by time.")
        clips.sort(key=lambda c: c["start"] if c["start"] is not None else c["path"])
        tracks[cam_id] = clips
    anchor_id = dirs[0][0]
    remaining = {cam: list(clips) for cam, clips in tracks.items() if cam != anchor_id}
    out = []
    for i, anchor in enumerate(tracks[anchor_id], 1):
        cams = {anchor_id: anchor}
        for cam, pool in remaining.items():
            if not pool:
                continue
            if anchor["start"] is None:
                idx = min(i - 1, len(pool) - 1)
                match = pool.pop(idx)
            else:
                match = min(pool, key=lambda c: abs(c["start"] - anchor["start"]))
                pool.remove(match)
            cams[cam] = match
        starts = [c["start"] for c in cams.values() if c["start"] is not None]
        if starts:
            s = max(starts)
            e = min(c["start"] + c["dur"] for c in cams.values() if c["start"] is not None)
            overlap = max(0.0, e - s)
        else:
            overlap = min(c["dur"] for c in cams.values())
        item = {
            "game": i,
            "cameras": [
                {"id": cam, "path": os.path.relpath(c["path"], root),
                 "start": c["start"], "dur": c["dur"]}
                for cam, c in cams.items()
            ],
            "overlap": overlap,
        }
        # Compatibility for older UI/caches.
        if "cam1" in cams:
            item["cam1"] = os.path.relpath(cams["cam1"]["path"], root)
            item["dur1"] = cams["cam1"]["dur"]
        if "cam2" in cams:
            item["cam2"] = os.path.relpath(cams["cam2"]["path"], root)
            item["dur2"] = cams["cam2"]["dur"]
        out.append(item)
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
        plan = json.load(f)
    if os.path.exists(os.path.join(STATE["root"], "net_rois.json")) and not plan.get("locate_enabled"):
        return None
    return plan


def _api_key(root):
    """API key for the ROI VLM judge: env first, else a local key file at
    data/_gui/anthropic_key.txt (gitignored). The key is loaded into the env for
    the anthropic client and NEVER logged or echoed anywhere."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    p = os.path.join(root, "data", "_gui", "anthropic_key.txt")
    if os.path.exists(p):
        with open(p) as f:
            key = f.read().strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            return True
    return False


def compute_plan(game, on_step=None):
    """Audio-only detection + planning for one game (no render). Returns plan dict."""
    from src.config import load_config
    from src.preprocess import preprocess_all
    from src.sync_engine import compute_offsets
    from src.peak_detector import detect_peaks
    from src.segment_planner import build_plan
    def step(i, n, label):
        if on_step:
            on_step(i, n, label)

    step(1, 7, "설정과 ROI 읽는 중")
    root = STATE["root"]
    cfg = load_config(os.path.join(root, "config.yaml"))
    rois_path = os.path.join(root, "net_rois.json")
    if os.path.exists(rois_path):
        cfg.locate.enabled = True
        cfg.locate.rois_path = rois_path
        # Real futsal audio often peaks several seconds after the ball is in the
        # net. When ROI is calibrated, search far enough before the cheer onset
        # to recover the actual goal frame instead of starting on celebration.
        cfg.locate.pre_sec = max(cfg.locate.pre_sec, cfg.build_up_sec + 11.0)
        cfg.locate.post_sec = max(cfg.locate.post_sec, 1.0)
    step(2, 7, "경기 파일 묶는 중")
    gl = games_list()
    g = next(x for x in gl if x["game"] == game)
    stage = os.path.join(root, "data", "_gui", f"game{game}")
    os.makedirs(stage, exist_ok=True)
    for name in os.listdir(stage):
        if os.path.splitext(name)[1].lower() in VIDEO_EXTS:
            os.remove(os.path.join(stage, name))
    staged_sources = {}
    for cam in g["cameras"]:
        rel = cam["path"]
        src = os.path.join(root, rel)
        ext = os.path.splitext(src)[1] or ".mp4"
        link = os.path.join(stage, f"{cam['id']}{ext.lower()}")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src), link)
        staged_sources[cam["id"]] = src
    step(3, 7, "카메라별 분석 오디오 추출 중")
    pre = preprocess_all(stage, os.path.join(root, "data", "_gui", "tv"),
                         os.path.join(root, "data", "_gui", f"ta{game}"),
                         cfg.fps, cfg.output_width, cfg.output_height)
    # point plan sources at the real originals (so /media can serve them for scrubbing)
    pre["source"] = {c: staged_sources[c] for c in pre["cams"]}
    camA = pre["cams"][0]
    step(4, 7, "카메라 싱크 계산 중")
    offsets = compute_offsets(pre["audio"], camA) if pre["is_multicam"] else {camA: 0.0}
    step(5, 7, "함성 피크 감지 중")
    peaks = detect_peaks(pre["audio"][camA], cfg)
    step(6, 7, "ROI 골망 모션과 앵글 구성 중")
    plan = build_plan(pre, offsets, peaks, camA, cfg)
    # rewrite src to root-relative so the browser can request /media/<rel>
    for clip in plan["clips"]:
        for seg in clip["segments"]:
            seg["src_rel"] = os.path.relpath(seg["src"], root)
    plan["game"] = game
    plan["offsets"] = offsets
    plan["camera_labels"] = {c: c for c in pre["cams"]}
    plan["locate_enabled"] = bool(cfg.locate.enabled and os.path.exists(cfg.locate.rois_path or ""))
    plan["locate_window"] = {"pre_sec": cfg.locate.pre_sec, "post_sec": cfg.locate.post_sec}
    # analysis WAV per cam -- the editor timeline draws its waveform from these
    plan["audio_rel"] = {c: os.path.relpath(p, root) for c, p in pre["audio"].items()}

    # Tier-1 verification of ROI-only (quiet-goal) candidates via the net-crop
    # VLM judge -- only when configured and an API key is available; otherwise
    # they stay flagged (roi_only) for manual review in the editor.
    if (cfg.locate.enabled and cfg.locate.scan_enabled
            and cfg.locate.scan_verify == "vlm" and _api_key(root)
            and any(c.get("roi_only") for c in plan["clips"])):
        step(7, 7, "ROI 후보 AI 판정 중")
        from src import goal_locator
        from src.goal_confirmer import confirm_roi_clips, make_net_classifier
        rois = goal_locator.load_rois(os.path.join(root, cfg.locate.rois_path))
        confirm_roi_clips(plan, cfg, rois, offsets, make_net_classifier(cfg),
                          os.path.join(root, "data", "_gui", "netcrops"))

    # Training-data flywheel for a future local goal/not-goal model: every
    # planned event is a labeled example (audio-confirmed vs ROI-only).
    try:
        import time
        with open(os.path.join(root, "data", "train_events.jsonl"), "a") as f:
            # kept clips AND judge-rejected ROI events (negatives matter most)
            for c in plan["clips"] + plan.get("roi_rejected", []):
                f.write(json.dumps({
                    "ts": int(time.time()), "game": game,
                    "cam": c.get("goal_cam") or c["segments"][0]["cam"],
                    "src": c["segments"][0].get("src_rel"),
                    "T": c["T"], "audio_confirmed": not c.get("roi_only"),
                    "scan_conf": c.get("scan_conf"),
                    "roi_verdict": c.get("roi_verdict"),
                }) + "\n")
    except OSError:
        pass
    plan.pop("roi_rejected", None)   # logged; keep the cached plan lean

    step(7, 7, "분석 결과 저장 중")
    with open(_plan_cache_path(game), "w") as f:
        json.dump(plan, f)
    return plan


def wave_env(rel, t0, t1, n):
    """Peak envelope of a WAV window, normalized 0..1, for timeline waveforms."""
    import numpy as np
    import soundfile as sf
    full = within_root(rel)
    info = sf.info(full)
    sr = info.samplerate
    a = max(0, int(t0 * sr))
    b = min(info.frames, int(t1 * sr))
    if b <= a:
        return []
    y = sf.read(full, start=a, stop=b, dtype="float32", always_2d=True)[0].mean(axis=1)
    n = max(10, min(int(n), 2000))
    idx = np.linspace(0, len(y), n + 1).astype(int)
    peaks = [float(np.abs(y[idx[i]:idx[i + 1]]).max()) if idx[i + 1] > idx[i] else 0.0
             for i in range(n)]
    m = max(peaks) or 1.0
    return [round(p / m, 3) for p in peaks]


JOBS = {}       # render job_id -> {"status": running|done|error, "progress": [i, n], ...}
PLAN_JOBS = {}  # plan job_id -> {"status": running|done|error, "progress": [i, n], ...}
PLAN_INFLIGHT = {}  # (root, game, fresh) -> running plan job_id
PLAN_LOCK = threading.Lock()


def _percent(i, n):
    return int(round(max(0, min(i, n)) / max(1, n) * 100))


def _plan_job(job_id, game, fresh):
    job = PLAN_JOBS[job_id]
    try:
        if not fresh:
            job.update(progress=[0, 1], percent=5, stage="캐시 확인 중")
            plan = cached_plan(game)
            if plan is not None:
                plan["cached"] = True
                job.update(status="done", progress=[1, 1], percent=100,
                           stage="캐시 로드 완료", plan=plan)
                return

        def on_step(i, n, label):
            job.update(progress=[i, n], percent=_percent(i, n), stage=label)

        plan = compute_plan(game, on_step=on_step)
        job.update(status="done", progress=[1, 1], percent=100,
                   stage="분석 완료", plan=plan)
    except Exception as e:  # noqa
        job.update(status="error", error=f"{type(e).__name__}: {e}",
                   stage="분석 실패")
    finally:
        key = job.get("key")
        if key is not None:
            with PLAN_LOCK:
                if PLAN_INFLIGHT.get(key) == job_id:
                    PLAN_INFLIGHT.pop(key, None)


def start_plan_job(game, fresh=False):
    key = (STATE["root"], int(game), bool(fresh))
    with PLAN_LOCK:
        existing = PLAN_INFLIGHT.get(key)
        if existing and PLAN_JOBS.get(existing, {}).get("status") == "running":
            return existing
        job_id = uuid.uuid4().hex[:12]
        PLAN_INFLIGHT[key] = job_id
        PLAN_JOBS[job_id] = {"status": "running", "progress": [0, 1],
                             "percent": 0, "stage": "대기 중", "key": key}
    t = threading.Thread(target=_plan_job, args=(job_id, game, fresh), daemon=True)
    t.start()
    return job_id


def _safe_output_name(name):
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", (name or "").strip()).strip(" .")
    if not name:
        name = "highlight_all.mp4"
    if os.path.splitext(name)[1].lower() != ".mp4":
        name += ".mp4"
    return name


def _render_job(job_id, plan, out_rel, bgm, bgm_volume, filename):
    from src.video_editor import render_plan
    root = STATE["root"]
    job = JOBS[job_id]
    work_dir = os.path.join(root, "data", "_gui", "render_jobs", job_id)
    try:
        final_dir = output_dir_path(out_rel)
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
        final_name = _safe_output_name(filename)
        final_path = os.path.join(final_dir, final_name)
        bgm_path = within_root(bgm) if bgm else None

        def on_clip(i, n, path):
            job.update(progress=[i, n], percent=_percent(i, max(1, n + 1)),
                       stage=f"클립 렌더링 {i}/{n}")

        job.update(percent=2, stage="렌더 준비 중")
        clips = render_plan(plan, work_dir, bgm_path, bgm_volume, on_clip=on_clip)
        src = os.path.join(work_dir, "highlight_all.mp4")
        if not os.path.exists(src):
            raise FileNotFoundError("highlight_all.mp4")
        job.update(percent=96, stage="최종 파일 이동 중")
        os.replace(src, final_path)
        shutil.rmtree(work_dir, ignore_errors=True)
        job.update(percent=100, stage="렌더 완료")
        job.update(status="done",
                   clip_count=len(clips), output=final_path)
    except Exception as e:  # noqa
        job.update(status="error", error=f"{type(e).__name__}: {e}")


def start_render(plan, out_rel, bgm=None, bgm_volume=0.15, filename=None):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "progress": [0, len(plan.get("clips", []))],
                    "percent": 0, "stage": "대기 중"}
    t = threading.Thread(target=_render_job,
                         args=(job_id, plan, out_rel, bgm, bgm_volume, filename), daemon=True)
    t.start()
    return job_id


def grab_frame(rel, t):
    full = within_root(rel)
    tmp = os.path.join(tempfile.gettempdir(), f"autopitch_frame_{abs(hash((rel, t)))}.jpg")
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
                return self._json(state_payload())
            if u.path == "/api/rois":
                p = os.path.join(STATE["root"], "net_rois.json")
                return self._json(json.load(open(p)) if os.path.exists(p) else {})
            if u.path == "/api/dirs":
                return self._json(list_dirs(urllib.parse.unquote(q.get("path", [""])[0])))
            if u.path == "/api/choose_dir":
                initial = urllib.parse.unquote(q.get("path", [""])[0])
                picked = choose_native_dir(output_dir_path(initial) if initial else STATE["root"])
                return self._json({"path": picked})
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
            if u.path == "/api/plan_status":
                job = PLAN_JOBS.get(q.get("job", [""])[0])
                return self._json(job if job else {"status": "error", "error": "unknown job"})
            if u.path == "/api/wave":
                return self._json({"peaks": wave_env(
                    urllib.parse.unquote(q["path"][0]),
                    float(q.get("t0", ["0"])[0]), float(q.get("t1", ["60"])[0]),
                    int(q.get("n", ["400"])[0]))})
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
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/api/bgm_upload":
                raw_name = q.get("name", ["bgm"])[0]
                name = os.path.basename(urllib.parse.unquote(raw_name))
                name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .") or "bgm"
                ext = os.path.splitext(name)[1].lower()
                if ext not in (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".mov", ".mp4"):
                    return self._err("unsupported audio file")
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0:
                    return self._err("empty upload")
                out_dir = os.path.join(STATE["root"], "data", "_gui", "bgm")
                os.makedirs(out_dir, exist_ok=True)
                out = os.path.join(out_dir, name)
                base, ext = os.path.splitext(out)
                k = 1
                while os.path.exists(out):
                    out = f"{base}_{k}{ext}"
                    k += 1
                with open(out, "wb") as f:
                    remaining = n
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                return self._json({"path": os.path.relpath(out, STATE["root"])})
            body = self._read_json()
            if u.path == "/api/root":
                p = os.path.realpath(os.path.expanduser(body["path"]))
                if not os.path.isdir(p):
                    return self._err("not a directory")
                STATE["root"] = p
                return self._json(state_payload())
            if u.path == "/api/prepare":
                cams = max(1, min(4, int(body.get("cams", 2))))
                raw = os.path.join(STATE["root"], "data", "raw")
                os.makedirs(raw, exist_ok=True)
                for i in range(1, cams + 1):
                    os.makedirs(os.path.join(raw, f"cam{i}"), exist_ok=True)
                return self._json(state_payload())
            if u.path == "/api/mkdir":
                parent = body.get("parent", "")
                name = re.sub(r"[^A-Za-z0-9._ -]+", "_", body.get("name", "")).strip(" .")
                if not name:
                    return self._err("folder name required")
                base = within_root(parent)
                out = os.path.realpath(os.path.join(base, name))
                root = os.path.realpath(STATE["root"])
                if out != root and not out.startswith(root + os.sep):
                    return self._err("path escapes project root")
                os.makedirs(out, exist_ok=True)
                return self._json(list_dirs(parent))
            if u.path == "/api/rois":
                json.dump(body["rois"], open(os.path.join(STATE["root"], "net_rois.json"), "w"), indent=2)
                return self._json({"ok": True})
            if u.path == "/api/plan_start":
                job = start_plan_job(int(body["game"]), bool(body.get("fresh")))
                return self._json({"job": job})
            if u.path == "/api/render":
                job = start_render(body["plan"], body.get("out", "data/output/gui"),
                                   body.get("bgm"), body.get("bgm_volume", 0.15),
                                   body.get("filename"))
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
    print(f"autoPitch GUI on http://127.0.0.1:{a.port}  (root={STATE['root']})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
