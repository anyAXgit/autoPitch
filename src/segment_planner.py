import numpy as np
from src.peak_detector import rms_db


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
    clips = []
    for T in peaks:
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
                    {"cam": camA, "src": pre["video"][camA],
                     "src_in": start, "src_out": cut},
                    {"cam": react_cam, "src": pre["video"][react_cam],
                     "src_in": max(0.0, cut + offsets[react_cam]),
                     "src_out": end + offsets[react_cam]},
                ]
            else:
                segments = [{"cam": camA, "src": pre["video"][camA],
                             "src_in": start, "src_out": end}]
        else:
            segments = [{"cam": camA, "src": pre["video"][camA],
                         "src_in": start, "src_out": end}]
        clips.append({"T": float(T), "segments": segments})
    return {"fps": cfg.fps, "crossfade_sec": cfg.crossfade_sec, "clips": clips}
