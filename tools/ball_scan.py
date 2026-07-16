"""Prototype: find a futsal ball resting in the net by classical CV (scipy only).

A goal leaves the ball briefly at rest inside the net -- bright, compact, roughly
circular, isolated. That's very different from a white jersey (large, elongated,
attached to a body) or net highlights (thin, stringy). We scan a dense window
around each ROI candidate, score every frame for the best ball-like bright blob,
and keep the top frames so a human (or a VLM) only looks where a ball plausibly is.

Run: ./.venv/bin/python tools/ball_scan.py <game> [--save DIR]
"""
import sys, os, json, subprocess
import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from src.config import load_config
from src import goal_locator as gl


def rgb_frames(source, t0, dur, roi, fps, margin=0.6):
    """Decode a window cropped to the net ROI (widened) as RGB frames + times."""
    x, y, w, h = roi
    x0 = max(0.0, x - w * margin); y0 = max(0.0, y - h * margin)
    x1 = min(1.0, x + w * (1 + margin)); y1 = min(1.0, y + h * (1 + margin))
    W = 256  # fixed decode width; height follows aspect
    vf = (f"crop=iw*{x1-x0:.4f}:ih*{y1-y0:.4f}:iw*{x0:.4f}:ih*{y0:.4f},"
          f"scale={W}:-2,fps={fps}")
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(max(0.0, t0)), "-i", source,
         "-t", str(dur), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    # infer height from byte count
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", source],
        capture_output=True, text=True).stdout
    ar = json.loads(probe)["streams"][0]
    H = int(round(W * (y1 - y0) * ar["height"] / ((x1 - x0) * ar["width"])))
    H -= H % 2
    n = out and len(out) // (W * H * 3)
    if not n:
        return np.empty((0, H, W, 3), np.uint8)
    return np.frombuffer(out, np.uint8)[:n * W * H * 3].reshape(n, H, W, 3)


def ball_score(frame):
    """Best ball-like bright blob in an RGB frame. Returns (score, (cx,cy,r))."""
    r, g, b = frame[..., 0].astype(int), frame[..., 1].astype(int), frame[..., 2].astype(int)
    bright = (r + g + b) / 3.0
    mn = np.minimum(np.minimum(r, g), b)
    mx = np.maximum(np.maximum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    # white-ish: bright AND low saturation (a colored jersey/post fails sat)
    mask = (bright > 175) & (sat < 0.28)
    if mask.sum() < 8:
        return 0.0, None
    lab, nlab = ndimage.label(mask)
    best, bestblob = 0.0, None
    H, W = mask.shape
    area_img = H * W
    for i in range(1, nlab + 1):
        ys, xs = np.where(lab == i)
        area = xs.size
        # ball at this scale ~ 40..900 px; below = noise, above = jersey/light
        if area < 30 or area > 0.06 * area_img:
            continue
        h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
        aspect = min(h, w) / max(h, w)          # disk ~1, string/jersey <<1
        extent = area / (h * w)                 # filled disk ~0.78
        if aspect < 0.55 or extent < 0.55:
            continue
        # isolation: bright ring just outside the blob should be dark (ball on net,
        # not a bright patch of a larger white region like a jersey)
        cy, cx = ys.mean(), xs.mean(); rad = (h + w) / 4.0
        yy, xx = np.ogrid[:H, :W]
        ring = (((yy - cy) ** 2 + (xx - cx) ** 2) >= (rad * 1.4) ** 2) & \
               (((yy - cy) ** 2 + (xx - cx) ** 2) <= (rad * 2.2) ** 2)
        iso = 1.0 - (mask[ring].mean() if ring.any() else 0.0)
        score = aspect * extent * iso * min(1.0, area / 200.0)
        if score > best:
            best, bestblob = score, (float(cx), float(cy), float(rad))
    return best, bestblob


def scan(game, save=None):
    cfg = load_config("config.yaml")
    rois = gl.load_rois("net_rois.json")
    p = json.load(open(f"data/_gui/plan_{game}.json"))
    off = p["offsets"]
    src = {}
    for c in p["clips"]:
        for g in c["segments"]:
            src.setdefault(g["cam"], g["src"])
    if save:
        os.makedirs(save, exist_ok=True)
    rows = []
    for c in p["clips"]:
        if not c.get("roi_only"):
            continue
        cam = c.get("goal_cam") or c["segments"][0]["cam"]
        roi = gl.roi_for_cam(cam, rois, src[cam])
        t = c["T"] + off.get(cam, 0.0)
        fr = rgb_frames(src[cam], t - 1.0, 4.5, roi, fps=6)   # dense window
        best, bestk, bestframe = 0.0, -1, None
        for k, f in enumerate(fr):
            s, blob = ball_score(f)
            if s > best:
                best, bestk, bestframe = s, k, f
        rows.append((c["T"], c.get("scan_conf"), cam, best))
        if save and bestframe is not None and best > 0:
            from PIL import Image
            Image.fromarray(bestframe).save(
                os.path.join(save, f"{c['T']:.0f}_{cam}_score{best:.2f}.png"))
    rows.sort(key=lambda r: -r[3])
    print(f"{'T':>8} {'conf':>5} {'cam':>5} {'ball_score':>10}")
    for T, conf, cam, s in rows:
        m = int(T // 60)
        print(f"{m:3}:{T%60:04.1f} {conf or 0:5.0f} {cam:>5} {s:10.3f}")


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "game3"
    save = None
    if "--save" in sys.argv:
        save = sys.argv[sys.argv.index("--save") + 1]
    scan(game, save)
