"""Net-ROI goal localization for FIXED cameras.

Because the cameras don't move, each goal's net sits at a fixed spot in the frame.
A ball hitting the net produces a sharp, localized motion spike there. We crop to a
calibrated net box, sample motion (frame-to-frame difference) in a short window
around the cheer onset, and take the biggest spike as the exact goal frame.

If no ROI is calibrated for a cam, or the spike isn't prominent enough (net not
visible / no clear disturbance), `locate_goal` returns None -> the caller falls
back to the audio cheer-onset anchor. This only ever *refines* timing; it never
drops a clip.
"""
import json
import os
import subprocess

import numpy as np


def load_rois(path):
    """Load net_rois.json: {camKeySubstring: [x, y, w, h]} in normalized (0-1)
    coords. Returns {} if path is falsy or missing."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def roi_for_cam(cam_id, rois):
    """Pick the ROI whose key is a (case-insensitive) substring of cam_id, so a
    box calibrated as "DJI" matches every DJI_* file across games. None if no match."""
    cl = cam_id.lower()
    for key, box in rois.items():
        if key.lower() in cl:
            return box
    return None


def _roi_gray_frames(source, t0, dur, roi, fps, px):
    """Decode a short window cropped to the net ROI as px*px grayscale frames.
    Returns an (n, px, px) float array. Decode-only (no re-encode)."""
    x, y, w, h = roi
    vf = (f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},fps={fps},"
          f"scale={px}:{px},format=gray")
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(max(0.0, t0)), "-i", source,
         "-t", str(dur), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    buf = np.frombuffer(out, dtype=np.uint8)
    n = buf.size // (px * px)
    return buf[:n * px * px].reshape(n, px, px).astype(np.float32)


def locate_goal(source, center_time, cfg, roi):
    """Find the goal frame near `center_time` (this cam's source timeline) via the
    net-motion spike inside `roi`. Returns {"goal_time", "confidence"} in the same
    (source) timeline, or None if there's no prominent spike (-> caller falls back).
    """
    lc = cfg.locate
    t0 = max(0.0, center_time - lc.pre_sec)
    dur = lc.pre_sec + lc.post_sec
    frames = _roi_gray_frames(source, t0, dur, roi, lc.fps, lc.frame_px)
    if len(frames) < 3:
        return None
    motion = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))   # (n-1,)
    times = t0 + (np.arange(len(motion)) + 0.5) / lc.fps
    i = int(np.argmax(motion))
    med = float(np.median(motion))
    mad = float(np.median(np.abs(motion - med))) or 1e-6
    prominence = (float(motion[i]) - med) / mad
    if prominence < lc.min_prominence:
        return None
    return {"goal_time": float(times[i]), "confidence": prominence}
