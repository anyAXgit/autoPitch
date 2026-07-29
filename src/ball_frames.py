"""Ball-guided evidence-frame selection for the net-crop VLM judge.

Classical CV here is an ATTENTION mechanism, not a gate: it scores every frame
in a dense window around the event for "a ball-like white blob inside the net
box" and picks the most promising moments. False positives (pad edges, jersey
patches through the mesh) are fine -- the VLM sees the frame and judges the
semantics. What CV must NOT do is miss the resting ball, so scoring is lenient.

Measured motivation: with 3-5 FIXED sample offsets the judge never saw the ball
(recall 1/5 on audio-confirmed goals); the ball rests in the net at a variable
moment within ~0-5s after impact, which dense scanning + selection catches.
"""
import json
import subprocess

import numpy as np
from scipy import ndimage
from src.ffmpeg import ffmpeg, ffprobe


def _decode_window(source, t0, dur, roi, fps=6.0, width=256, margin=0.35):
    """RGB frames of the widened net crop + the tight net box in crop pixels."""
    x, y, w, h = roi
    x0 = max(0.0, x - w * margin); y0 = max(0.0, y - h * margin)
    x1 = min(1.0, x + w * (1 + margin)); y1 = min(1.0, y + h * (1 + margin))
    vf = (f"crop=iw*{x1-x0:.5f}:ih*{y1-y0:.5f}:iw*{x0:.5f}:ih*{y0:.5f},"
          f"scale={width}:-2,fps={fps}")
    out = subprocess.run(
        [ffmpeg(), "-v", "error", "-ss", str(max(0.0, t0)), "-i", source,
         "-t", str(dur), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    probe = subprocess.run(
        [ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", source],
        capture_output=True, text=True).stdout
    ar = json.loads(probe)["streams"][0]
    H = int(round(width * (y1 - y0) * ar["height"] / ((x1 - x0) * ar["width"])))
    H -= H % 2
    n = len(out) // (width * H * 3) if H else 0
    frames = (np.frombuffer(out, np.uint8)[:n * width * H * 3].reshape(n, H, width, 3)
              if n else np.empty((0, max(H, 2), width, 3), np.uint8))
    cw, ch = (x1 - x0), (y1 - y0)
    tight = (int((x - x0) / cw * width), int((y - y0) / ch * H),
             int(w / cw * width), int(h / ch * H))
    return frames, tight


def ball_score(frame, tight, bg=None):
    """How ball-like is the best white blob inside the tight net box (0..1)."""
    tx, ty, tw, th = tight
    r, g, b = (frame[..., c].astype(int) for c in range(3))
    bright = (r + g + b) / 3.0
    mn = np.minimum(np.minimum(r, g), b); mx = np.maximum(np.maximum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    mask = (bright > 170) & (sat < 0.30)
    if bg is not None:
        mask &= np.abs(frame.astype(int) - bg).mean(axis=2) > 25
    gate = np.zeros_like(mask)
    gate[max(0, ty):ty + th, max(0, tx):tx + tw] = True
    mask &= gate
    if mask.sum() < 8:
        return 0.0
    lab, nlab = ndimage.label(mask)
    best = 0.0
    for i in range(1, nlab + 1):
        ys, xs = np.where(lab == i)
        area = xs.size
        if area < 20 or area > 0.35 * tw * th:
            continue
        hh = ys.max() - ys.min() + 1; ww = xs.max() - xs.min() + 1
        aspect = min(hh, ww) / max(hh, ww); extent = area / (hh * ww)
        if aspect < 0.5 or extent < 0.5:
            continue
        best = max(best, aspect * extent * min(1.0, area / 150.0))
    return float(best)


def select_ball_times(source, t, roi, pre=1.0, post=5.5, fps=6.0, top=3):
    """Times (source timeline) of the frames most likely to show a ball in the
    net around event `t`. Returns up to `top` times, spread at least 0.5s apart;
    empty list if nothing ball-like was seen (caller falls back to fixed offsets)."""
    frames, tight = _decode_window(source, t - pre, pre + post, roi, fps=fps)
    if not len(frames):
        return []
    bg = np.median(frames, axis=0)
    scores = [ball_score(f, tight, bg) for f in frames]
    order = np.argsort(scores)[::-1]
    picked = []
    for i in order:
        if scores[i] <= 0.2:
            break
        ti = (t - pre) + (i + 0.5) / fps
        if all(abs(ti - p) >= 0.5 for p in picked):
            picked.append(float(ti))
        if len(picked) >= top:
            break
    return sorted(picked)
