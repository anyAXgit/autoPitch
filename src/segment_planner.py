import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from src.peak_detector import rms_db
from src import goal_locator


def reaction_end(times, db, T, cfg, peak=None):
    """First time after the cheer that stays quiet for hold_sec, clamped to
    [min_len, max_len] measured from T-build_up.

    "After the cheer", not "after T": the net-ROI moves the anchor back to the
    goal frame, which can be many seconds ahead of the crowd, and the gap
    between them is quiet by definition. Searching from T found that gap and
    closed the clip inside it -- so a clip proposed by a cheer could end before
    its own cheer. Measured on one game: 10 of 23 hand-edited clips left under
    3s after the peak, and the person pushed 7 of those ends out by a median of
    5s. The worst had its cheer 4.9s past the end (17:46, cheer at 18:00, clip
    ending 17:56) -- the goal itself was off the back of the clip.
    """
    start = max(0.0, T - cfg.build_up_sec)
    # Nothing before the moment that created this clip can count as "settled".
    floor = T if peak is None else max(T, float(peak))
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
        if t <= floor:
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


def _pool(cfg, n):
    """Threads for concurrent ffmpeg probes, never more than there is work."""
    return max(1, min(int(getattr(cfg.locate, "workers", 6)), n, (os.cpu_count() or 2)))


def _probe_nets(pre, offsets, onsets, cfg, rois, progress=None, watch=None):
    """`locate_goal` for every (onset, camera) pair, run concurrently.

    Each probe is one ffmpeg process seeking to its own window, so they do not
    depend on each other and the loop that used to run them one at a time was
    just waiting. Seeking really is the right shape here -- decoding the whole
    match at the ROI once measured 159s against 30s for 66 windows -- so the win
    is in overlapping the seeks, not avoiding them. ffmpeg threads its own decode,
    which caps this near 1.7x rather than scaling with cores.
    """
    jobs = []
    for cam in pre["cams"]:
        roi = goal_locator.roi_for_cam(cam, rois, pre["source"].get(cam))
        if not roi:
            continue
        for onset in onsets:
            jobs.append((onset, cam, roi))
    out, done = {}, 0
    if not jobs:
        return out
    def probe(job):
        onset, cam, roi = job
        src = pre["source"][cam]
        patch = {}
        # Only ask for the patch when someone is watching, so the plain call
        # shape stays what it was.
        kw = {"on_patch": lambda b, px: patch.update(gray=b, px=px)} if watch else {}
        r = goal_locator.locate_goal(src, onset + offsets.get(cam, 0.0), cfg, roi, **kw)
        # A still only for the ones that hit. Decoding one per probe would undo
        # the concurrency this stage just bought; a hit is ~1 in 3, and it is the
        # only one anybody wants to look at.
        still = (goal_locator.still_jpeg(src, r["goal_time"])
                 if watch and r else None)
        return r, patch, still

    # Walked in submission order, not completion order, so the position the
    # caller shows moves forward through the match instead of jumping around.
    with ThreadPoolExecutor(max_workers=_pool(cfg, len(jobs))) as ex:
        for (onset, cam, _roi), (r, patch, still) in zip(jobs, ex.map(probe, jobs)):
            out[(onset, cam)] = r
            done += 1
            if watch:
                watch({"at": onset, "cam": cam, "hit": bool(r),
                       "conf": (r["confidence"] if r else None),
                       "goal_time": (r["goal_time"] - offsets.get(cam, 0.0)) if r else None,
                       "patch": patch or None, "still": still})
            if progress:
                progress(done, len(jobs), onset, cam)
    return out


def _refine_anchor(pre, offsets, onset, cfg, rois, probes=None):
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
        r = (probes.get((onset, cam)) if probes is not None
             else goal_locator.locate_goal(pre["source"][cam], onset + off, cfg, roi))
        if r:
            candidates.append((r["goal_time"] - off, cam, r["confidence"]))
    if not candidates:
        return None
    # A widened ROI search window recovers delayed cheers, but it can also include
    # keeper movement or players brushing the net long before the goal. Keep
    # candidates that are plausibly tied to this cheer; if all candidates are too
    # early, leave the audio onset alone instead of over-refining to noise.
    max_lead = cfg.locate.max_lead_sec
    candidates = [c for c in candidates if c[0] >= onset - max_lead and c[0] <= onset + cfg.locate.post_sec]
    if not candidates:
        return None
    # If the earliest hit is far ahead of the cheer but another camera has a
    # strong hit right around the onset, the early one is often stale net/keeper
    # motion from the previous phase. Keep the "earliest" rule for normal delayed
    # cheers, but do not let a very early candidate steal a near-onset goal hit.
    near_post = max(cfg.locate.post_sec, 1.0)
    near = [c for c in candidates if onset - 1.25 <= c[0] <= onset + near_post]
    earliest = min(candidates, key=lambda x: x[0])
    if near and earliest[0] < onset - 6.0:
        best = max(near, key=lambda x: x[2])
        return (best[0], best[1])
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
    return [(T, peak, cam, True, conf) for T, peak, cam, conf in kept]


def _local_audio_peaks(times, db, threshold, min_gap_sec):
    """Return local dB maxima above `threshold`, gap-suppressed by loudness."""
    candidates = []
    for i in range(1, len(db) - 1):
        if db[i] >= threshold and db[i] >= db[i - 1] and db[i] >= db[i + 1]:
            candidates.append((float(times[i]), float(db[i])))
    if not candidates:
        return []
    kept = []
    for t, loud in candidates:
        if kept and t - kept[-1][0] < min_gap_sec:
            if loud > kept[-1][1]:
                kept[-1] = (t, loud)
        else:
            kept.append((t, loud))
    return kept


def _weak_audio_roi_anchors(pre, offsets, cfg, rois, existing, rms_cache, camA, progress=None):
    """Add candidates when a non-anchor camera has a sub-threshold cheer and its
    ROI confirms a net hit.

    This is deliberately narrower than full ROI-only scanning: it only probes
    short windows around local audio peaks, so quiet goals can be recovered
    without letting every net-area motion event become a clip.
    """
    candidates = []
    weak_k = getattr(cfg.locate, "weak_audio_k", 0.0)
    if weak_k <= 0:
        return []
    jobs = []
    for cam in pre["cams"]:
        if cam == camA:
            continue
        roi = goal_locator.roi_for_cam(cam, rois, pre["source"].get(cam))
        if not roi or cam not in rms_cache:
            continue
        times, db = rms_cache[cam]
        threshold = float(np.median(db) + weak_k * np.std(db))
        off = offsets.get(cam, 0.0)
        local = [p for p in _local_audio_peaks(times, db, threshold, cfg.min_gap_sec)
                 if p[0] - off >= 0]
        jobs.extend((cam, roi, off, peak_src, loud) for peak_src, loud in local)

    # This is the most expensive stage per clip it recovers: one ffmpeg decode
    # each, and on the game measured 69 probes cost 40s and produced exactly one
    # candidate. Spend the budget on the loudest, which is where the yield was --
    # that one candidate was the loudest of the 69. `weak_audio_max_probes` is
    # the cap; raise it to trade minutes for recall.
    cap = int(getattr(cfg.locate, "weak_audio_max_probes", 0) or len(jobs))
    if len(jobs) > cap:
        jobs = sorted(jobs, key=lambda j: -j[4])[:cap]
    jobs.sort(key=lambda j: j[3])
    total, done = len(jobs), 0
    if not jobs:
        return []

    def probe(job):
        cam, roi, _off, peak_src, _loud = job
        return goal_locator.locate_goal(pre["source"][cam], peak_src, cfg, roi)

    with ThreadPoolExecutor(max_workers=_pool(cfg, total)) as ex:
        for (cam, roi, off, peak_src, _loud), r in zip(jobs, ex.map(probe, jobs)):
            done += 1
            if progress:
                mm = f"{int(peak_src // 60)}:{int(peak_src % 60):02d}"
                progress(f"약한 오디오 후보 ROI 확인 {done}/{total} · {cam} {mm}",
                         done, max(1, total))
            if not r:
                continue
            if r["confidence"] < getattr(cfg.locate, "weak_audio_min_confidence", 0.0):
                continue
            peak_ref = peak_src - off
            T = r["goal_time"] - off
            max_lead = getattr(cfg.locate, "weak_audio_max_lead_sec", cfg.build_up_sec)
            if T < peak_ref - max_lead or T > peak_ref + cfg.locate.post_sec:
                continue
            candidates.append((T, peak_ref, cam, r["confidence"]))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    kept = []
    for cand in candidates:
        if kept and cand[0] - kept[-1][0] < cfg.min_gap_sec:
            if (cand[0], -cand[3]) < (kept[-1][0], -kept[-1][3]):
                kept[-1] = cand
        else:
            kept.append(cand)
    return [(T, peak, cam, True, conf) for T, peak, cam, conf in kept]


def _dedupe_anchors(anchors, cfg, camA):
    """Drop only near-identical anchors.

    Weak-audio+ROI anchors are intentionally allowed to sit close to an audio
    anchor. In real matches those can be distinct phases of play, and merging
    them loses recall. Full ROI-only scan candidates are already suppressed near
    existing audio anchors before this point.
    """
    if not anchors:
        return []
    anchors = sorted(anchors, key=lambda a: a[0])
    out = []
    for anchor in anchors:
        if not out or anchor[0] - out[-1][0] >= 1.0:
            out.append(anchor)
            continue
        cur = out[-1]
        if cur[3] and anchor[3] and (anchor[4] or 0.0) > (cur[4] or 0.0):
            out[-1] = anchor
        # Otherwise keep the earlier/audio-backed anchor.
    return out


def _mean_db(times, db, a, b):
    mask = (times >= a) & (times < b)
    return float(np.mean(db[mask])) if mask.any() else -np.inf


def _max_db_on_ref_times(ref_times, rms_cache, offsets):
    """Return a crowd-loudness envelope on the reference timeline using the
    loudest available camera at each timestamp. This keeps celebrations from
    ending early just because the main/goal camera got quiet first."""
    stacked = []
    for cam, (times, db) in rms_cache.items():
        shifted = ref_times + offsets.get(cam, 0.0)
        vals = np.interp(shifted, times, db, left=-np.inf, right=-np.inf)
        stacked.append(vals)
    return np.max(np.vstack(stacked), axis=0) if stacked else np.array([])


def k_cache_from(rms_cache):
    """Each camera's loudness restated in standard deviations above its own
    median.

    Raw dB cannot be compared between cameras: a phone and an action cam at
    opposite corners have different gain, different distance to the crowd and
    different noise floors, so "louder in dB" can just mean "hotter preamp".
    Comparing k asks the question that was meant -- which camera is this moment
    unusual for -- and that is the one nearest the ball.
    """
    out = {}
    for cam, (times, db) in rms_cache.items():
        sd = float(np.std(db))
        out[cam] = (times, (db - float(np.median(db))) / sd if sd > 0
                    else np.zeros_like(db))
    return out


def _loudest_cam(cams, offsets, lo, hi, k_cache, exclude=None):
    """Camera that heard [lo, hi] most, in k. None when there is no candidate."""
    best, best_k = None, -np.inf
    for cam in cams:
        if cam == exclude or cam not in k_cache:
            continue
        cam_times, cam_k = k_cache[cam]
        m = _mean_db(cam_times, cam_k,
                     max(0.0, lo + offsets.get(cam, 0.0)), hi + offsets.get(cam, 0.0))
        if m > best_k:
            best, best_k = cam, m
    return best


def _label_for(goal_labels, peak):
    """Vision verdict for a peak, tolerating float drift. DATA ONLY -- this is
    never the render flag; burning a badge stays the editor's manual choice."""
    if not goal_labels:
        return None
    p = float(peak)
    if p in goal_labels:
        return goal_labels[p]
    for k, v in goal_labels.items():
        if abs(k - p) < 0.01:
            return v
    return None


def build_plan(pre, offsets, peaks, camA, cfg, progress=None, goal_labels=None,
               watch=None):
    def report(label, frac):
        if progress:
            progress(label, max(0.0, min(1.0, float(frac))))

    report("카메라 오디오 RMS 분석 중", 0.02)
    times, db = rms_db(pre["audio"][camA], cfg.rms_window_sec)
    rms_cache = {camA: (times, db)}
    if pre["is_multicam"]:
        others = [cam for cam in pre["cams"] if cam != camA]
        for i, cam in enumerate(others, 1):
            if cam == camA:
                continue
            report(f"카메라 오디오 RMS 분석 중 {i + 1}/{len(others) + 1}", 0.02 + 0.06 * i / max(1, len(others)))
            rms_cache[cam] = rms_db(pre["audio"][cam], cfg.rms_window_sec)
    k_cache = k_cache_from(rms_cache)
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
    # Onsets are pure arithmetic on the RMS envelope, so resolve them all up
    # front; that turns the net probes into one independent batch that can run
    # concurrently instead of one ffmpeg call at a time down the loop.
    onsets = [cheer_onset(times, db, peak, cfg) for peak in peaks]
    probes = {}
    if rois:
        def probe_progress(done, total, at, cam):
            report(f"ROI 골망 모션 확인 {done}/{total} · {cam} "
                   f"{int(at // 60)}:{int(at % 60):02d}",
                   0.10 + 0.55 * done / max(1, total))
        probes = _probe_nets(pre, offsets, onsets, cfg, rois,
                             progress=probe_progress, watch=watch)
    for idx, peak in enumerate(peaks):
        # Anchor timing on the cheer ONSET (goal moment), not the loudness peak,
        # so the buildup reliably contains the shot instead of starting mid-celebration.
        T = onsets[idx]
        goal_cam = None
        if rois:
            # Fixed-camera net-ROI motion pinpoints the exact goal frame AND which
            # cam's net was hit (= the goal-side camera). Fall back to onset / audio
            # angle when the net isn't visible / no clear disturbance.
            refined = _refine_anchor(pre, offsets, T, cfg, rois, probes)
            if refined is not None:
                T, goal_cam = refined
        anchors.append((T, peak, goal_cam, False, None))
    report(f"오디오 피크 ROI 확인 완료 {len(peaks)}/{len(peaks)}", 0.65)
    if rois:
        def weak_progress(label, done, total):
            report(label, 0.66 + 0.24 * done / max(1, total))
        anchors.extend(_weak_audio_roi_anchors(pre, offsets, cfg, rois, anchors, rms_cache, camA,
                                               progress=weak_progress))
    if rois and cfg.locate.scan_enabled:
        report("전체 ROI-only 후보 스캔 중", 0.90)
        anchors.extend(_roi_only_anchors(pre, offsets, cfg, rois, anchors))
    report("하이라이트 컷과 앵글 구성 중", 0.94)
    anchors = _dedupe_anchors(anchors, cfg, camA)

    prev_end = 0.0
    for idx, (T, peak, goal_cam, roi_only, scan_conf) in enumerate(anchors):
        # Never open before the previous clip closed. Anchors are not evenly
        # spaced by the time they get here: `_refine_anchor` moves one to the
        # net-motion spike, up to max_lead (11s) ahead of its cheer, so two
        # anchors that cleared min_gap as peaks can end up seconds apart. Only
        # the END was clamped, and its floor (T + post_goal + min_reaction) wins
        # when that happens -- so the next clip's buildup reached back across it
        # and both clips showed the same seconds. Measured on one game: 12 pairs
        # overlapping, the worst sharing 8 of its 10 seconds. Losing some buildup
        # is the cheaper mistake; the same footage twice reads as a bug.
        start = max(0.0, T - cfg.build_up_sec, prev_end)
        hi = start + cfg.max_len_sec
        if idx + 1 < len(anchors):
            next_start = anchors[idx + 1][0] - cfg.build_up_sec
            # never end past the next clip's start (keep a sane floor so the
            # reaction doesn't collapse when goals are absurdly close)
            hi = min(hi, max(next_start, T + cfg.post_goal_sec + cfg.min_reaction_sec))
        end_db = _max_db_on_ref_times(times, rms_cache, offsets) if pre["is_multicam"] else db
        base_end = reaction_end(times, end_db, T, cfg, peak)
        min_end = start + cfg.min_len_sec
        # Tail is useful after a real, sustained celebration, but adding it to
        # every minimum-length clip makes dead time drag. Only append it when
        # audio actually kept the clip past the minimum.
        tail = cfg.tail_sec if base_end > min_end + cfg.rms_window_sec else 0.0
        end = min(base_end + tail, hi)
        if pre["is_multicam"]:
            # PRIMARY angle (buildup + the goal itself) = the goal-side cam when the
            # net-ROI identified it, else Cam A. This shows the goal from the camera
            # nearest to where it was scored instead of whichever cam is loudest.
            # Which camera shows the goal, in order of how much we actually know:
            # Which camera shows the goal. The net-ROI knows which net the ball
            # hit, but that turned out to be the wrong question to ask it: the
            # two ROIs do not measure comparable things. Each camera sits by one
            # goal, so its own net fills the frame and the keeper standing in it
            # moves more pixels than a ball entering the far net ever does. On
            # five clips checked by eye the near-net camera won on motion every
            # time and was wrong every time; loudness named the right camera in
            # all five. So ROI refines WHEN the goal happened, and sound decides
            # WHERE to look. Set `locate.angle_from_roi` to put it back -- with a
            # tightly-drawn ROI that excludes the keeper it may well be better.
            if cfg.locate.angle_from_roi and goal_cam and goal_cam in pre["cams"]:
                primary = goal_cam
            elif cfg.main_cam:
                primary = camA          # an explicit instruction outranks a guess
            else:
                # Judge the footage that will be shown -- the whole clip, not
                # [T, end]. `T` is the ROI-refined anchor, so a window starting
                # there moves when the ROI moves: at 10:28 of the game measured
                # here it had walked far enough forward that the cheer which
                # proposed the clip fell outside the window entirely, and the
                # vote went to the camera with the higher noise floor. Handing
                # the angle back to the ROI through the back door is exactly
                # what deciding by sound was meant to stop.
                primary = _loudest_cam(pre["cams"], offsets, start, end, k_cache) or camA
            # Hold the primary cam a beat past the goal before any switch, so the
            # ball settling into the net and the first beat of celebration play
            # on the goal-side cam. Grow `end` if needed so a reaction segment
            # would keep at least min_reaction_sec.
            end = min(max(end, T + cfg.post_goal_sec + cfg.min_reaction_sec), hi)
            cut = max(T, min(T + cfg.post_goal_sec, end - cfg.min_reaction_sec))
            # Then ask the same question the primary angle was decided by, over
            # the stretch a second angle would occupy: who heard this? Switching
            # used to mean "cut to the other camera", which with two cameras is
            # not a choice at all -- there is only one other. Measured across two
            # games, three of the four switches cut to the QUIETER camera, one of
            # them away from a +4.70 celebration to a +0.72 empty half. A cut has
            # to earn itself; if the camera already on screen is still the one
            # hearing it, stay.
            react = (_loudest_cam(pre["cams"], offsets, cut, end, k_cache)
                     if end - cut >= cfg.min_angle_switch_sec else None)
            segments = ([seg(primary, start, cut), seg(react, cut, end)]
                        if react and react != primary else [seg(primary, start, end)])
        else:
            segments = [seg(camA, start, end)]
        # T = goal-onset anchor (clip start / label); peak = loudness peak; goal_cam = net-ROI goal side
        verdict = _label_for(goal_labels, peak)
        clips.append({"T": float(T), "peak": float(peak),
                      "goal_cam": goal_cam, "roi_only": bool(roi_only),
                      "scan_conf": (float(scan_conf) if scan_conf is not None else None),
                      # goal_label = burn a GOAL badge. Manual, editor-only: an
                      # audio candidate may not be a goal, so we never caption
                      # the video from a model verdict.
                      "goal_label": False,
                      # vision_* = the model's verdict, recorded as data for
                      # review / sorting / future training. Never drawn.
                      "vision_goal": (verdict["is_goal"] if verdict else None),
                      "vision_conf": (verdict["confidence"] if verdict else None),
                      "segments": segments})
        prev_end = end
    report(f"하이라이트 {len(clips)}개 구성 완료", 1.0)
    return {"fps": cfg.fps, "crossfade_sec": cfg.crossfade_sec,
            "output_width": pre["width"], "output_height": pre["height"],
            "hw_encode": cfg.hw_encode, "video_crf": cfg.video_crf,
            "video_preset": cfg.video_preset, "clips": clips}
