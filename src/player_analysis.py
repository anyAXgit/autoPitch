"""EXPERIMENTAL -- team-level movement analysis. Not wired into the pipeline
or the editor: measured on real footage it is not trustworthy enough to ship.

Why it is shelved (all measured, 7/16 footage):
  * Detection works -- 11 players found per frame, 99-117px tall.
  * Per-player tracking does NOT -- 11 players fragmented into 109 track IDs
    over 90s, so distance-run and per-player heatmaps are out.
  * Detection COUNTS are camera-visibility, not team activity: the two corner
    cameras each over-count whichever team is nearer, which flipped A/B totals
    between cam1 (4786/3527) and cam2 (3805/4265) for the same window.
  * Cameras cannot be fused without a court homography, and three automatic
    calibration attempts failed (line detection, near-goal PnP, far-goal PnP --
    the last verified wrong by back-projection).
  * Team split by shirt colour is exposure-dependent: cam2 renders the dark
    team at luminance 124 vs cam1's 57, narrowing the gap to the white team.

A single camera overlooking the whole court removes most of these at once.
Kept for that day; see docs for the full measurement log.

Team-level movement analysis from the fixed cameras.

Per-PLAYER tracking is not reliable at this camera geometry -- measured on real
footage, 11 players fragmented into 109 track IDs over 90s (a track breaks every
~2.8s) because the oblique low angle makes players occlude each other and both
teams wear uniform kits. So this module deliberately avoids identity: it uses
per-frame DETECTIONS only, splits them into two teams by shirt colour, and
reports things that do not need identity -- occupancy heatmaps, territory share
and activity spread.

Detection needs `ultralytics` (AGPL-3.0); it is an optional extra, imported
lazily so the core pipeline never depends on it.
"""
import json
import os
import subprocess

import numpy as np
from src.ffmpeg import ffmpeg

GRID_H, GRID_W = 36, 64          # heatmap resolution (image space)



def homography(src4, dst4):
    """4-point DLT homography (numpy only -- no OpenCV dependency).

    `src4` are image pixels (court corners as marked in the editor), `dst4` the
    matching court-plane points in metres. Positions are only converted when the
    user has actually calibrated: an oblique corner view compresses depth so
    badly that an *assumed* court quad would produce confidently wrong territory
    numbers, which is worse than reporting none.
    """
    A = []
    for (x, y), (u, v) in zip(src4, dst4):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def to_court(pts, H):
    """Apply a homography to Nx2 image points -> court metres."""
    p = np.hstack([np.asarray(pts, dtype=np.float64), np.ones((len(pts), 1))])
    q = p @ H.T
    return q[:, :2] / q[:, 2:3]


def _frames(source, start, dur, fps, width, out_dir):
    """Decode a window to JPGs (decode-only, no re-encode)."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(max(0.0, start)), "-i", source, "-t", str(dur),
                    "-vf", f"fps={fps},scale={width}:-2", "-an",
                    os.path.join(out_dir, "f_%05d.jpg")], check=True)
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if f.endswith(".jpg"))


def _shirt_colour(img, box):
    """Median RGB of the torso band -- the upper-middle of the person box.

    Head and legs are excluded: hair and turf would drag the colour toward the
    background, which is exactly what has to stay out of the team split.
    """
    x0, y0, x1, y1 = [int(v) for v in box]
    h = max(1, y1 - y0)
    ty0 = y0 + int(h * 0.18)
    ty1 = y0 + int(h * 0.48)
    tx0 = x0 + int((x1 - x0) * 0.25)
    tx1 = x0 + int((x1 - x0) * 0.75)
    crop = img[max(0, ty0):max(1, ty1), max(0, tx0):max(1, tx1)]
    if crop.size == 0:
        return np.zeros(3, dtype=np.float32)
    return np.median(crop.reshape(-1, crop.shape[-1]), axis=0).astype(np.float32)


def _split_teams(colours):
    """2-means on shirt colour. Returns labels (0/1) and the two centroids.

    Deterministic seeding (darkest / brightest sample) keeps the label->team
    mapping stable across runs so saved analyses stay comparable.
    """
    c = np.asarray(colours, dtype=np.float32)
    if len(c) < 2:
        return np.zeros(len(c), dtype=int), c[:1].repeat(2, axis=0) if len(c) else np.zeros((2, 3))
    lum = c.sum(axis=1)
    cent = np.stack([c[int(np.argmin(lum))], c[int(np.argmax(lum))]])
    for _ in range(12):
        d = np.linalg.norm(c[:, None, :] - cent[None, :, :], axis=2)
        lab = np.argmin(d, axis=1)
        for k in (0, 1):
            if np.any(lab == k):
                cent[k] = c[lab == k].mean(axis=0)
    return lab, cent


def analyse(source, start=0.0, dur=120.0, fps=5, width=1280,
            model_name="yolov8n.pt", conf=0.35, work_dir="build/tmp/analysis",
            progress=None, court_quad=None, court_size=(20.0, 12.0)):
    """Detect players over a window and return team-level occupancy stats.

    Returns a dict with heatmaps (per team + combined), territory share, spread
    and the frame size -- everything in IMAGE space. A court homography can be
    applied later without redoing detection.
    """
    from ultralytics import YOLO                      # lazy: AGPL optional dep

    fdir = os.path.join(work_dir, "frames")
    if os.path.isdir(fdir):
        for f in os.listdir(fdir):
            os.remove(os.path.join(fdir, f))
    files = _frames(source, start, dur, fps, width, fdir)
    if not files:
        raise RuntimeError(f"no frames decoded from {source}")

    model = YOLO(model_name)
    pts, cols = [], []
    for i, fp in enumerate(files):
        if progress and i % 20 == 0:
            progress(f"선수 검출 {i}/{len(files)}", i / max(1, len(files)))
        res = model(fp, imgsz=width, classes=[0], conf=conf, verbose=False)[0]
        if res.boxes is None or not len(res.boxes):
            continue
        img = res.orig_img[:, :, ::-1]                # BGR -> RGB
        for b in res.boxes.xyxy.tolist():
            pts.append(((b[0] + b[2]) / 2.0, b[3]))   # feet = ground contact
            cols.append(_shirt_colour(img, b))
    if not pts:
        raise RuntimeError("no players detected")

    H, W = res.orig_shape if hasattr(res, "orig_shape") else (720, width)
    pts = np.asarray(pts, dtype=np.float32)
    labels, cent = _split_teams(cols)

    def heat(mask):
        if not np.any(mask):
            return np.zeros((GRID_H, GRID_W))
        h, _, _ = np.histogram2d(pts[mask, 1], pts[mask, 0],
                                 bins=[GRID_H, GRID_W], range=[[0, H], [0, W]])
        return h

    hm_a, hm_b = heat(labels == 0), heat(labels == 1)
    # Territory needs real court coordinates. Without a calibrated quad the
    # image midpoint is NOT the halfway line (perspective), so report nothing
    # rather than a plausible-looking wrong number.
    court = None
    half = {"A": None, "B": None}
    if court_quad is not None and len(court_quad) == 4:
        CW, CH = court_size
        H_ = homography(court_quad, [(0, 0), (CW, 0), (CW, CH), (0, CH)])
        court = to_court(pts, H_)
        inside = (court[:, 0] >= -1) & (court[:, 0] <= CW + 1) & \
                 (court[:, 1] >= -1) & (court[:, 1] <= CH + 1)
        for k, v in (("A", 0), ("B", 1)):
            m = (labels == v) & inside
            half[k] = float(np.mean(court[m, 0] < CW / 2)) if np.any(m) else None

    def spread(mask):
        """Std of positions = how wide the team ranges (px)."""
        if np.count_nonzero(mask) < 2:
            return 0.0
        return float(np.hypot(pts[mask, 0].std(), pts[mask, 1].std()))

    return {
        "source": os.path.basename(source), "start": float(start), "dur": float(dur),
        "fps": fps, "frames": len(files), "detections": int(len(pts)),
        "width": int(W), "height": int(H),
        "teams": {
            "A": {"colour": [float(x) for x in cent[0]], "n": int(np.sum(labels == 0)),
                  "own_half_share": half["A"], "spread_px": spread(labels == 0)},
            "B": {"colour": [float(x) for x in cent[1]], "n": int(np.sum(labels == 1)),
                  "own_half_share": half["B"], "spread_px": spread(labels == 1)},
        },
        "calibrated": court is not None,
        "court_size": list(court_size) if court is not None else None,
        "heatmap": {"A": hm_a.tolist(), "B": hm_b.tolist(),
                    "all": (hm_a + hm_b).tolist(), "grid": [GRID_H, GRID_W]},
        "court_pts": ({"A": court[labels == 0].tolist(),
                       "B": court[labels == 1].tolist()} if court is not None else None),
    }



def merge(results, court_size=(20.0, 12.0)):
    """Fuse per-camera analyses into one.

    Only meaningful when every camera was calibrated: each view's heatmap lives
    in its own image space, so pooling them without a court homography would add
    apples to oranges. When they ARE calibrated the fusion is worth having --
    the two corner cameras cover each other's blind side, so a player hidden
    behind a cluster in one view is usually clear in the other.

    Falls back to returning the inputs untouched (side-by-side, not merged).
    """
    if not results:
        return None
    if not all(r.get("calibrated") and r.get("court_pts") for r in results):
        return {"merged": False, "reason": "코트 미보정 카메라가 있어 융합하지 않음",
                "per_camera": results}
    CW, CH = court_size
    grid_h, grid_w = results[0]["heatmap"]["grid"]
    out = {"A": np.zeros((grid_h, grid_w)), "B": np.zeros((grid_h, grid_w))}
    teams = {k: {"n": 0} for k in ("A", "B")}
    half = {"A": [], "B": []}
    for r in results:
        for k in ("A", "B"):
            pts = np.asarray(r["court_pts"][k], dtype=float)
            if len(pts) == 0:
                continue
            h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[grid_h, grid_w],
                                     range=[[0, CH], [0, CW]])
            out[k] += h
            teams[k]["n"] += len(pts)
            half[k].append(float(np.mean(pts[:, 0] < CW / 2)))
    for k in ("A", "B"):
        teams[k]["own_half_share"] = float(np.mean(half[k])) if half[k] else None
        teams[k]["spread_px"] = float(np.mean([r["teams"][k]["spread_px"] for r in results]))
    return {"merged": True, "cameras": len(results), "calibrated": True,
            "court_size": [CW, CH], "teams": teams,
            "detections": sum(r["detections"] for r in results),
            "frames": sum(r["frames"] for r in results),
            "dur": max(r["dur"] for r in results),
            "heatmap": {"A": out["A"].tolist(), "B": out["B"].tolist(),
                        "all": (out["A"] + out["B"]).tolist(),
                        "grid": [grid_h, grid_w]}}


def save(result, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return path
