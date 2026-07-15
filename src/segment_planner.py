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
    each cam's calibrated ROI (fixed cameras). Returns `(goal_time_camA, goal_cam)`
    -- the refined time AND the cam whose net was hit (= the goal-side camera) --
    or None if no cam has a prominent spike (caller keeps the onset / audio angle)."""
    candidates = []
    for cam in pre["cams"]:
        roi = goal_locator.roi_for_cam(cam, rois, pre["source"].get(cam))
        if not roi:
            continue
        off = offsets.get(cam, 0.0)
        r = goal_locator.locate_goal(pre["source"][cam], onset + off, cfg, roi)
        if r:
            candidates.append((r["goal_time"] - off, cam, r["confidence"]))
    if not candidates:
        return None
    # A widened ROI search window recovers delayed cheers, but it can also include
    # keeper movement or players brushing the net long before the goal. Keep
    # candidates that are plausibly tied to this cheer; if all candidates are too
    # early, leave the audio onset alone instead of over-refining to noise.
    max_lead = max(cfg.build_up_sec + 6.0, 11.0)
    candidates = [c for c in candidates if c[0] >= onset - max_lead and c[0] <= onset + cfg.locate.post_sec]
    if not candidates:
        return None
    # The ball hitting the net normally precedes the crowd peak/onset, and the
    # closer camera can have lower ROI prominence than a wider camera with larger
    # net/player motion. Prefer the earliest valid net hit over raw confidence so
    # the buildup starts before the goal-side angle.
    best = min(candidates, key=lambda x: (x[0], -x[2]))
    return (best[0], best[1]) if best else None


def _roi_only_anchors(pre, offsets, cfg, rois, existing):
    """Return ROI-only anchors `(T, peak, goal_cam)` that are not already covered
    by audio-derived anchors. These raise recall for quiet goals, while min-gap
    suppression keeps them from duplicating normal audio clips."""
    candidates = []
    for cam in pre["cams"]:
        roi = goal_locator.roi_for_cam(cam, rois, pre["source"].get(cam))
        if not roi:
            continue
        off = offsets.get(cam, 0.0)
        for ev in goal_locator.scan_goal_events(pre["source"][cam], cfg, roi, cfg.min_gap_sec,
                                                cache_path=cfg.locate.scan_cache):
            T = ev["goal_time"] - off
            if T < 0:
                continue
            if any(abs(T - anchor[0]) < cfg.min_gap_sec for anchor in existing):
                continue
            candidates.append((T, T, cam, ev["confidence"]))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    kept = []
    for cand in candidates:
        if kept and cand[0] - kept[-1][0] < cfg.min_gap_sec:
            # If two cameras see the same quiet goal, use the earliest ROI hit.
            if (cand[0], -cand[3]) < (kept[-1][0], -kept[-1][3]):
                kept[-1] = cand
        else:
            kept.append(cand)
    return [(T, peak, cam, True) for T, peak, cam, _conf in kept]


def _mean_db(times, db, a, b):
    mask = (times >= a) & (times < b)
    return float(np.mean(db[mask])) if mask.any() else -np.inf


def _loudest_other(pre, offsets, exclude, T, end, cfg, rms_cache):
    """Loudest cam over the celebration window [T, end], excluding `exclude`.
    Returns None if there's no other cam."""
    best, best_db = None, -np.inf
    for cam in pre["cams"]:
        if cam == exclude:
            continue
        a = max(0.0, T + offsets.get(cam, 0.0))
        b = end + offsets.get(cam, 0.0)
        cam_times, cam_db = rms_cache[cam]
        m = _mean_db(cam_times, cam_db, a, b)
        if m > best_db:
            best, best_db = cam, m
    return best


def pick_reaction_angle(pre, offsets, camA, T, end, cfg, rms_cache):
    if not pre["is_multicam"]:
        return camA
    return _loudest_other(pre, offsets, camA, T, end, cfg, rms_cache) or camA


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
    def seg(cam, a, b):
        off = offsets.get(cam, 0.0)
        return {"cam": cam, "src": pre["source"][cam],
                "src_in": max(0.0, a + off), "src_out": b + off}

    # Resolve every clip's anchor first so each clip can be clamped against the
    # NEXT one: min_gap (15s) < max_len (25s), so back-to-back goals could
    # otherwise overlap and duplicate the same scene in consecutive clips.
    anchors = []
    for peak in peaks:
        # Anchor timing on the cheer ONSET (goal moment), not the loudness peak,
        # so the buildup reliably contains the shot instead of starting mid-celebration.
        T = cheer_onset(times, db, peak, cfg)
        goal_cam = None
        if rois:
            # Fixed-camera net-ROI motion pinpoints the exact goal frame AND which
            # cam's net was hit (= the goal-side camera). Fall back to onset / audio
            # angle when the net isn't visible / no clear disturbance.
            refined = _refine_anchor(pre, offsets, T, cfg, rois)
            if refined is not None:
                T, goal_cam = refined
        anchors.append((T, peak, goal_cam, False))
    if rois and cfg.locate.scan_enabled:
        anchors.extend(_roi_only_anchors(pre, offsets, cfg, rois, anchors))
    anchors.sort(key=lambda a: a[0])   # refinement can nudge order
    # When two cheers merge into one continuous roar, both onsets can collapse
    # to the same rise -- that's one goal moment, not two near-identical clips.
    deduped = []
    for a in anchors:
        if deduped and a[0] - deduped[-1][0] < 1.0:
            continue
        deduped.append(a)
    anchors = deduped

    for idx, (T, peak, goal_cam, roi_only) in enumerate(anchors):
        start = max(0.0, T - cfg.build_up_sec)
        hi = start + cfg.max_len_sec
        if idx + 1 < len(anchors):
            next_start = anchors[idx + 1][0] - cfg.build_up_sec
            # never end past the next clip's start (keep a sane floor so the
            # reaction doesn't collapse when goals are absurdly close)
            hi = min(hi, max(next_start, T + cfg.post_goal_sec + cfg.min_reaction_sec))
        end = min(reaction_end(times, db, T, cfg) + cfg.tail_sec, hi)
        if pre["is_multicam"]:
            # PRIMARY angle (buildup + the goal itself) = the goal-side cam when the
            # net-ROI identified it, else Cam A. This shows the goal from the camera
            # nearest to where it was scored instead of whichever cam is loudest.
            primary = goal_cam if (goal_cam and goal_cam in pre["cams"]) else camA
            react = _loudest_other(pre, offsets, primary, T, end, cfg, rms_cache)
            if react is not None:
                # Hold the primary cam a beat past the goal before switching to the
                # reaction angle, so the ball settling into the net and the first
                # beat of celebration play on the goal-side cam. Grow `end` if needed
                # so the reaction segment keeps at least min_reaction_sec.
                end = min(max(end, T + cfg.post_goal_sec + cfg.min_reaction_sec), hi)
                cut = max(T, min(T + cfg.post_goal_sec, end - cfg.min_reaction_sec))
                segments = [seg(primary, start, cut), seg(react, cut, end)]
            else:
                segments = [seg(primary, start, end)]
        else:
            segments = [seg(camA, start, end)]
        # T = goal-onset anchor (clip start / label); peak = loudness peak; goal_cam = net-ROI goal side
        clips.append({"T": float(T), "peak": float(peak),
                      "goal_cam": goal_cam, "roi_only": bool(roi_only),
                      "segments": segments})
    return {"fps": cfg.fps, "crossfade_sec": cfg.crossfade_sec,
            "output_width": pre["width"], "output_height": pre["height"],
            "hw_encode": cfg.hw_encode, "clips": clips}
