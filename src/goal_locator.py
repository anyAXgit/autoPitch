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


def roi_for_cam(cam_id, rois, source_path=None):
    """Pick the ROI for a camera/source.

    New GUI calibration stores per-file keys such as
    `data/raw/cam1/DJI_...MP4`, because the same cam folder can contain games
    with different framing. Match those exactly/with path suffix first.

    Legacy keys are still supported for device/file-name substrings like `DJI`
    or `IMG`, but broad `cam1`/`cam2` keys are intentionally not applied to every
    source when `source_path` is known.
    """
    if source_path:
        src = source_path.replace("\\", "/").lower()
        base = os.path.basename(src)
        for key, box in rois.items():
            k = key.replace("\\", "/").lower()
            if src.endswith(k) or base == os.path.basename(k):
                return box
        for key, box in rois.items():
            k = key.lower()
            if not k.startswith("cam") and k in base:
                return box
        return None
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


def _roi_gray_frames_full(source, roi, fps, px):
    """Decode the whole source cropped to the net ROI."""
    x, y, w, h = roi
    vf = (f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},fps={fps},"
          f"scale={px}:{px},format=gray")
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", source,
         "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    buf = np.frombuffer(out, dtype=np.uint8)
    n = buf.size // (px * px)
    return buf[:n * px * px].reshape(n, px, px).astype(np.float32)


def _roi_gray_keyframes(source, roi, px):
    x, y, w, h = roi
    vf = f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale={px}:{px},format=gray"
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", source,
         "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    buf = np.frombuffer(out, dtype=np.uint8)
    n = buf.size // (px * px)
    return buf[:n * px * px].reshape(n, px, px).astype(np.float32)


def _motion_events(frames, t0, fps, min_prominence, min_gap_sec=0.0, frame_times=None):
    if len(frames) < 3:
        return []
    motion = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))   # (n-1,)
    if frame_times is not None and len(frame_times) >= len(frames):
        times = np.asarray(frame_times[:len(frames)], dtype=np.float32)[1:]
    else:
        times = t0 + (np.arange(len(motion)) + 0.5) / fps
    med = float(np.median(motion))
    mad = float(np.median(np.abs(motion - med))) or 1e-6
    prom = (motion - med) / mad
    active = np.where(prom >= min_prominence)[0]
    events = []
    if active.size:
        start = prev = int(active[0])
        for raw_i in active[1:]:
            i = int(raw_i)
            if i == prev + 1:
                prev = i
            else:
                peak_i = start + int(np.argmax(prom[start:prev + 1]))
                events.append({"goal_time": float(times[start]), "confidence": float(prom[peak_i])})
                start = prev = i
        peak_i = start + int(np.argmax(prom[start:prev + 1]))
        events.append({"goal_time": float(times[start]), "confidence": float(prom[peak_i])})
    if min_gap_sec <= 0 or len(events) <= 1:
        return events
    kept = []
    for ev in events:
        if kept and ev["goal_time"] - kept[-1]["goal_time"] < min_gap_sec:
            if ev["confidence"] > kept[-1]["confidence"]:
                kept[-1] = ev
        else:
            kept.append(ev)
    return kept


def event_impulse_ok(source, event_time, cfg, roi):
    """Tier-0 goal/junk gate on an ROI event's TEMPORAL SHAPE (free, no ML).

    A ball hitting the net is a brief impulse (<~1s of high motion, then quick
    decay); a keeper leaning on the net, players brushing past, or someone
    fetching the ball is broad sustained motion. Measure how long motion stays
    above half the event's peak inside a +-2s window at the fine fps: impulses
    pass, blobs fail.
    """
    lc = cfg.locate
    t0 = max(0.0, event_time - 2.0)
    frames = _roi_gray_frames(source, t0, 4.0, roi, lc.fps, lc.frame_px)
    if len(frames) < 5:
        return True   # can't judge -> don't drop
    motion = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))
    med = float(np.median(motion))
    peak = float(motion.max())
    if peak <= med:
        return False
    half = med + (peak - med) * 0.5
    active_sec = float(np.count_nonzero(motion >= half)) / lc.fps
    return active_sec <= lc.scan_max_impulse_sec


def locate_goal(source, center_time, cfg, roi):
    """Find the goal frame near `center_time` (this cam's source timeline) via the
    net-motion spike inside `roi`. Returns {"goal_time", "confidence"} in the same
    (source) timeline, or None if there's no prominent spike (-> caller falls back).
    """
    lc = cfg.locate
    t0 = max(0.0, center_time - lc.pre_sec)
    dur = lc.pre_sec + lc.post_sec
    frames = _roi_gray_frames(source, t0, dur, roi, lc.fps, lc.frame_px)
    events = _motion_events(frames, t0, lc.fps, lc.min_prominence)
    if not events:
        return None
    return max(events, key=lambda ev: ev["confidence"])


def _scan_cache_key(source, roi, lc):
    try:
        mtime = int(os.path.getmtime(source))
    except OSError:
        mtime = 0
    return json.dumps([os.path.abspath(source), mtime, [round(v, 4) for v in roi],
                       lc.scan_min_prominence, lc.scan_fps, lc.scan_frame_px,
                       lc.scan_max_impulse_sec], sort_keys=True)


def scan_goal_events(source, cfg, roi, min_gap_sec, cache_path=None):
    """Find ROI-only candidate goals across the whole source.

    This is stricter than `locate_goal`: it can create new clip candidates without
    an audio peak, so it uses `scan_min_prominence`, min-gap suppression, and the
    impulse-shape gate (`event_impulse_ok`) to reject sustained non-goal motion
    (keeper on the net, passers-by). Results are cached per (source, roi, params)
    so re-planning doesn't re-decode the whole match.
    """
    lc = cfg.locate
    cache_path = cache_path or lc.scan_cache
    key = _scan_cache_key(source, roi, lc)
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}
    if key in cache:
        return cache[key]

    scan_px = getattr(lc, "scan_frame_px", min(lc.frame_px, 32))
    frames = _roi_gray_keyframes(source, roi, scan_px)
    if len(frames) >= 3:
        rough = _motion_events(frames, 0.0, 1.0, lc.scan_min_prominence, min_gap_sec)
    else:
        scan_fps = getattr(lc, "scan_fps", min(lc.fps, 2.0))
        frames = _roi_gray_frames_full(source, roi, scan_fps, scan_px)
        rough = _motion_events(frames, 0.0, scan_fps, lc.scan_min_prominence, min_gap_sec)

    refined = []
    for ev in rough:
        r = locate_goal(source, ev["goal_time"], cfg, roi)
        ev = r or ev
        if event_impulse_ok(source, ev["goal_time"], cfg, roi):   # Tier-0 shape gate
            refined.append(ev)

    if cache_path:
        cache[key] = refined
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return refined
