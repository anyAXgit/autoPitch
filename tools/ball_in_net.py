"""Prototype: detect a futsal ball AT REST INSIDE the calibrated net (goal cue).

Two ideas the earlier measurements pointed to:
  1) Geometry gate -- search only inside the tight net ROI (the calibration box),
     so a ball sitting on the field beside the goal is excluded automatically.
  2) Persistence -- after a goal the ball rests in the net for ~1-3s, so it shows
     up in MANY consecutive frames at a stable spot; a passing person or a motion
     artifact appears for a frame or two. We require a stable ball across a run of
     frames, which kills the one-off false positives the motion scan drowns in.

For each ROI-only candidate we densely sample the window, find the best in-net
ball blob per frame, and score by the longest stable run. Output ranks candidates
by that score and saves an annotated evidence frame (net box + detected ball).

Run: ./.venv/bin/python tools/ball_in_net.py <game> [--save DIR]
"""
import sys, os, json, subprocess
import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from src.config import load_config
from src import goal_locator as gl


def decode(source, t0, dur, roi, fps, margin):
    """Decode [t0, t0+dur] cropped to the net ROI (+margin) as RGB frames, and
    return (frames, tight_box_px) where tight_box_px is the calibration ROI's
    pixel rect inside the returned crop (so we can gate on 'ball inside net')."""
    x, y, w, h = roi
    x0 = max(0.0, x - w * margin); y0 = max(0.0, y - h * margin)
    x1 = min(1.0, x + w * (1 + margin)); y1 = min(1.0, y + h * (1 + margin))
    W = 256
    vf = (f"crop=iw*{x1-x0:.5f}:ih*{y1-y0:.5f}:iw*{x0:.5f}:ih*{y0:.5f},"
          f"scale={W}:-2,fps={fps}")
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(max(0.0, t0)), "-i", source,
         "-t", str(dur), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", source],
        capture_output=True, text=True).stdout
    ar = json.loads(probe)["streams"][0]
    H = int(round(W * (y1 - y0) * ar["height"] / ((x1 - x0) * ar["width"]))); H -= H % 2
    n = len(out) // (W * H * 3) if H else 0
    frames = (np.frombuffer(out, np.uint8)[:n * W * H * 3].reshape(n, H, W, 3)
              if n else np.empty((0, H, W, 3), np.uint8))
    cw, ch = (x1 - x0), (y1 - y0)
    tight = (int((x - x0) / cw * W), int((y - y0) / ch * H),
             int(w / cw * W), int(h / ch * H))     # (px_x, px_y, px_w, px_h)
    return frames, tight


def find_ball(frame, tight):
    """Best ball-like white blob whose centroid is INSIDE the tight net box.
    Returns (score, (cx, cy, r)) or (0, None)."""
    tx, ty, tw, th = tight
    r, g, b = (frame[..., c].astype(int) for c in range(3))
    bright = (r + g + b) / 3.0
    mn = np.minimum(np.minimum(r, g), b); mx = np.maximum(np.maximum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    mask = (bright > 175) & (sat < 0.28)
    # gate: only inside the net box
    gate = np.zeros_like(mask)
    gate[max(0, ty):ty + th, max(0, tx):tx + tw] = True
    mask &= gate
    if mask.sum() < 8:
        return 0.0, None
    lab, nlab = ndimage.label(mask)
    H, W = mask.shape
    best, blob = 0.0, None
    for i in range(1, nlab + 1):
        ys, xs = np.where(lab == i)
        area = xs.size
        if area < 25 or area > 0.4 * tw * th:
            continue
        hh = ys.max() - ys.min() + 1; ww = xs.max() - xs.min() + 1
        aspect = min(hh, ww) / max(hh, ww); extent = area / (hh * ww)
        if aspect < 0.55 or extent < 0.55:
            continue
        cy, cx = ys.mean(), xs.mean(); rad = (hh + ww) / 4.0
        score = aspect * extent * min(1.0, area / 200.0)
        if score > best:
            best, blob = score, (float(cx), float(cy), float(rad))
    return best, blob


def analyze(frames, tight, min_q=0.35):
    """Longest run of consecutive frames holding a ball at a stable location."""
    hits = [find_ball(f, tight) for f in frames]
    run, best_run, best_center = 0, 0, None
    prev = None
    for q, blob in hits:
        ok = q >= min_q and blob is not None
        if ok and prev is not None and np.hypot(blob[0]-prev[0], blob[1]-prev[1]) <= max(6, blob[2]*1.5):
            run += 1
        elif ok:
            run = 1
        else:
            run = 0
        prev = blob if ok else None
        if run > best_run:
            best_run, best_center = run, blob
    # best single frame for evidence
    bi = int(np.argmax([q for q, _ in hits])) if hits else -1
    return best_run, (hits[bi] if bi >= 0 else (0, None)), bi


def draw(frame, tight, blob):
    im = frame.copy()
    tx, ty, tw, th = tight
    for xx in (tx, tx + tw):
        if 0 <= xx < im.shape[1]: im[max(0,ty):ty+th, xx] = [80, 160, 255]
    for yy in (ty, ty + th):
        if 0 <= yy < im.shape[0]: im[yy, max(0,tx):tx+tw] = [80, 160, 255]
    if blob:
        cx, cy, r = blob
        yy, xx = np.ogrid[:im.shape[0], :im.shape[1]]
        ring = np.abs(np.hypot(xx - cx, yy - cy) - r*1.3) < 1.5
        im[ring] = [255, 40, 40]
    return im


def run(game, save=None):
    cfg = load_config("config.yaml")
    rois = gl.load_rois("net_rois.json")
    p = json.load(open(f"data/_gui/plan_{game}.json"))
    off = p["offsets"]
    src = {}
    for c in p["clips"]:
        for g in c["segments"]:
            src.setdefault(g["cam"], g["src"])
    if save: os.makedirs(save, exist_ok=True)
    rows = []
    for c in p["clips"]:
        if not c.get("roi_only"):
            continue
        cam = c.get("goal_cam") or c["segments"][0]["cam"]
        roi = gl.roi_for_cam(cam, rois, src[cam])
        t = c["T"] + off.get(cam, 0.0)
        frames, tight = decode(src[cam], t - 1.0, 4.5, roi, fps=6, margin=0.12)
        best_run, (q, blob), bi = analyze(frames, tight)
        rows.append((c["T"], c.get("scan_conf"), cam, best_run, q))
        if save and blob is not None:
            from PIL import Image
            Image.fromarray(draw(frames[bi], tight, blob)).save(
                os.path.join(save, f"{c['T']:.0f}_{cam}_run{best_run}_q{q:.2f}.png"))
    rows.sort(key=lambda r: (-r[3], -r[4]))
    print(f"{'T':>8} {'str':>4} {'cam':>5} {'ball_run':>8} {'quality':>7}   verdict")
    for T, conf, cam, run_, q in rows:
        v = "BALL-IN-NET ●" if run_ >= 3 else ("weak" if run_ >= 2 else "—")
        print(f"{int(T//60):3}:{T%60:04.1f} {conf or 0:4.0f} {cam:>5} {run_:8} {q:7.2f}   {v}")


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "game3"
    save = sys.argv[sys.argv.index("--save")+1] if "--save" in sys.argv else None
    run(game, save)
