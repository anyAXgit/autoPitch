import numpy as np
from src.peak_detector import rms_db
from src import goal_locator


def reaction_end(times, db, T, cfg):
    """First time after T that stays quiet for hold_sec, clamped to
    [min_len, max_len] measured from T-build_up."""
    start = max(0.0, T - cfg.build_up_sec)
    lo = start + cfg.min_len_sec
    hi = start + cfg.max_len_sec
    # "Settled" must be measured against the pre-goal ambient, not the
    # whole-track median: celebration audio can dominate (or merely sit
    # close to) the track median, which would make the crowd's own
    # cheering look "quiet" and collapse the clip to min_len. Use the
    # buildup window [start, T] as the quiet baseline instead.
    pre_mask = (times >= start) & (times <= T)
    pre_baseline = np.median(db[pre_mask]) if np.any(pre_mask) else np.median(db)
    quiet_level = pre_baseline + cfg.margin_db   # below this = crowd settled
    hold_needed = cfg.hold_sec
    quiet_run_start = None
    end = hi
    for t, d in zip(times, db):
        if t <= T:
            continue
        if d < quiet_level:
            if quiet_run_start is None:
                quiet_run_start = t
            elif t - quiet_run_start >= hold_needed:
                end = quiet_run_start
                break
        else:
            quiet_run_start = None
    return float(min(max(end, lo), hi))


def cheer_onset(times, db, peak, cfg):
    """Anchor the clip at the RISING EDGE of the cheer, not its loudness peak.

    The crowd peaks a variable ~1-3s AFTER the ball crosses the line, so anchoring
    the clip on the loudness peak can push the actual goal off the front of the
    clip ("celebration with no goal"). Walk backward from the peak to the frame
    where dB first rose above the pre-cheer ambient (`baseline + margin_db`) — that
    onset sits right on the goal moment. `peak` is still used elsewhere for angle
    loudness; this only moves the *timing* anchor earlier.
    """
    p = int(np.argmin(np.abs(times - peak)))
    look = cfg.build_up_sec + 2.0
    pre = db[(times >= max(0.0, peak - look)) & (times < peak)]
    # low percentile, not median: the cheer often fills part of this window, so a
    # median would ride up with it. p20 recovers the pre-cheer ambient floor.
    baseline = float(np.percentile(pre, 20)) if pre.size else float(np.median(db))
    # Anchor at the FOOT of the rise (baseline + 2dB), not where it hits the
    # +margin cheer level -- by +margin the cheer is already well up and the goal
    # is a beat behind. Walking back from a real loudness peak stays inside the
    # contiguous cheer, so a small foot threshold doesn't catch stray murmurs.
    level = baseline + 2.0
    i = p
    while i > 0 and db[i] > level:
        i -= 1
    onset = float(times[min(i + 1, p)])
    return min(onset, float(peak))


def _refine_anchor(pre, offsets, onset, cfg, rois):
    """Refine the onset anchor to the exact goal frame via the net-motion spike in
    each cam's calibrated ROI (fixed cameras). Returns a camA-timeline time, or None
    if no cam has a prominent spike (caller keeps the onset -- goal not visible)."""
    best = None
    for cam in pre["cams"]:
        roi = goal_locator.roi_for_cam(cam, rois)
        if not roi:
            continue
        off = offsets.get(cam, 0.0)
        r = goal_locator.locate_goal(pre["source"][cam], onset + off, cfg, roi)
        if r and (best is None or r["confidence"] > best[1]):
            best = (r["goal_time"] - off, r["confidence"])   # back to camA timeline
    return best[0] if best else None


def _mean_db(times, db, a, b):
    mask = (times >= a) & (times < b)
    return float(np.mean(db[mask])) if mask.any() else -np.inf


def pick_reaction_angle(pre, offsets, camA, T, end, cfg, rms_cache):
    if not pre["is_multicam"]:
        return camA
    best, best_db = camA, -np.inf
    for cam in pre["cams"]:
        if cam == camA:
            continue
        a = max(0.0, T + offsets[cam])
        b = end + offsets[cam]
        cam_times, cam_db = rms_cache[cam]
        m = _mean_db(cam_times, cam_db, a, b)
        if m > best_db:
            best, best_db = cam, m
    return best


def build_plan(pre, offsets, peaks, camA, cfg):
    times, db = rms_db(pre["audio"][camA], cfg.rms_window_sec)
    rms_cache = {camA: (times, db)}
    if pre["is_multicam"]:
        for cam in pre["cams"]:
            if cam == camA:
                continue
            rms_cache[cam] = rms_db(pre["audio"][cam], cfg.rms_window_sec)
    rois = goal_locator.load_rois(cfg.locate.rois_path) if cfg.locate.enabled else {}
    clips = []
    for peak in peaks:
        # Anchor timing on the cheer ONSET (goal moment), not the loudness peak,
        # so the buildup reliably contains the shot instead of starting mid-celebration.
        T = cheer_onset(times, db, peak, cfg)
        if rois:
            # Fixed-camera net-ROI motion pinpoints the exact goal frame; fall back
            # to onset when the net isn't visible / no clear disturbance.
            refined = _refine_anchor(pre, offsets, T, cfg, rois)
            if refined is not None:
                T = refined
        start = max(0.0, T - cfg.build_up_sec)
        hi = start + cfg.max_len_sec
        end = min(reaction_end(times, db, T, cfg) + cfg.tail_sec, hi)
        if pre["is_multicam"]:
            react_cam = pick_reaction_angle(pre, offsets, camA, T, end, cfg, rms_cache)
            if react_cam != camA:
                # Hold Cam A a beat past the goal peak before switching to the
                # reaction angle (goal peak T = loudest cheer, slightly after
                # the ball crosses the line), so the ball settling into the net
                # and the first beat of celebration play on the main cam. Grow
                # `end` if needed so the reaction segment keeps at least
                # min_reaction_sec, clamped under max_len.
                end = min(max(end, T + cfg.post_goal_sec + cfg.min_reaction_sec), hi)
                cut = max(T, min(T + cfg.post_goal_sec, end - cfg.min_reaction_sec))
                segments = [
                    {"cam": camA, "src": pre["source"][camA],
                     "src_in": start, "src_out": cut},
                    {"cam": react_cam, "src": pre["source"][react_cam],
                     "src_in": max(0.0, cut + offsets[react_cam]),
                     "src_out": end + offsets[react_cam]},
                ]
            else:
                segments = [{"cam": camA, "src": pre["source"][camA],
                             "src_in": start, "src_out": end}]
        else:
            segments = [{"cam": camA, "src": pre["source"][camA],
                         "src_in": start, "src_out": end}]
        # T = goal-onset anchor (clip start / label); peak = loudness peak (kept for reference)
        clips.append({"T": float(T), "peak": float(peak), "segments": segments})
    return {"fps": cfg.fps, "crossfade_sec": cfg.crossfade_sec,
            "output_width": pre["width"], "output_height": pre["height"],
            "clips": clips}
