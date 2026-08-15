import numpy as np

from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.config import Config, load_config
from src.segment_planner import build_plan, reaction_end, cheer_onset
from src.peak_detector import detect_peaks
from src.sync_engine import compute_offsets


def _cfg(tmp_path, **peak):
    import yaml
    d = {"build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 20,
         "peak": {"min_gap_sec": 5, "threshold_k": 2.0}}
    d["peak"].update(peak)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(d))
    return load_config(str(p))


def test_single_cam_plan_one_segment(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 12)]},
    ], duration=50.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, {"camA": 0.0}, peaks, "camA", cfg)
    clip = plan["clips"][0]
    assert len(clip["segments"]) == 1
    seg = clip["segments"][0]
    assert seg["cam"] == "camA"
    assert abs(seg["src_in"] - (clip["T"] - 5)) < 0.6      # build_up before T
    length = seg["src_out"] - seg["src_in"]
    assert 8 <= length <= 20                               # clamped dynamic length


def test_tail_not_added_to_minimum_length_clip(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 31, 12)]},
    ], duration=50.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg_yaml(tmp_path, reaction={"tail_sec": 4})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = build_plan(res, {"camA": 0.0}, peaks, "camA", cfg)["clips"][0]
    length = clip["segments"][0]["src_out"] - clip["segments"][0]["src_in"]
    assert length <= cfg.min_len_sec + 0.6


def test_tail_added_after_sustained_celebration(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 44, 12)]},
    ], duration=60.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg_yaml(tmp_path, reaction={"tail_sec": 4})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = build_plan(res, {"camA": 0.0}, peaks, "camA", cfg)["clips"][0]
    length = clip["segments"][0]["src_out"] - clip["segments"][0]["src_in"]
    assert length > cfg.min_len_sec + 2


def _mean_k(audio_path, offset, a, b, cfg):
    """Mean loudness of [a, b] on the reference clock, in this recording's own
    standard deviations. Derived here from the audio rather than by calling the
    planner's helper, so the test checks the choice instead of restating it."""
    import numpy as np
    from src.peak_detector import rms_db as _rms
    t, db = _rms(audio_path, cfg.rms_window_sec)
    k = (db - np.median(db)) / (np.std(db) or 1.0)
    m = (t >= a + offset) & (t < b + offset)
    return float(np.mean(k[m])) if m.any() else float("-inf")


def _by_loudness(res, offsets, lo, hi, cfg):
    """Cameras ordered loudest-first over [lo, hi]."""
    return sorted(res["audio"], key=lambda c: -_mean_k(res["audio"][c], offsets.get(c, 0.0), lo, hi, cfg))


def test_multicam_reaction_uses_louder_subcam(tmp_path):
    raw = tmp_path / "raw"
    # camB reaction burst (gain 18) is louder than camC (gain 8) at T~30
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 18)]},
        {"name": "camC", "color": "blue", "offset": 0.0, "bursts": [(30, 34, 8)]},
    ], duration=50.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    # Without a net-ROI answer the buildup goes to whichever camera heard the
    # moment most, and the reaction to the loudest of the rest. Which camera that
    # is comes from the audio, not from the generator's gain numbers -- each
    # recording has its own spread, so a bigger gain is not automatically a
    # bigger k.
    start = min(g["src_in"] - offsets.get(g["cam"], 0.0) for g in clip["segments"])
    end = max(g["src_out"] - offsets.get(g["cam"], 0.0) for g in clip["segments"])
    T = clip["T"]
    cut = max(T, min(T + cfg.post_goal_sec, end - cfg.min_reaction_sec))

    # The two angles answer different questions and so listen to different
    # windows: the buildup cam is the one that heard the clip being shown, the
    # reaction cam the one that heard the celebration.
    assert clip["segments"][0]["cam"] == _by_loudness(res, offsets, start, end, cfg)[0]

    # Assert the RULE, not a segment count. Whether these three synthetic bursts
    # hand the tail to a second camera comes down to k differences of a few
    # hundredths, and those move with the platform's audio encoder -- this used
    # to pass on macOS and fail on Linux for exactly that reason. A switch that
    # has to earn itself does not happen on demand, so ask whether the right
    # thing happened either way.
    tail_leader = _by_loudness(res, offsets, cut, end, cfg)[0]
    if tail_leader == clip["segments"][0]["cam"]:
        assert len(clip["segments"]) == 1, "이미 나온 카메라로 넘기면 안 된다"
    else:
        assert len(clip["segments"]) == 2
        assert clip["segments"][1]["cam"] == tail_leader


def test_multicam_end_uses_loudest_camera_not_only_main_cam(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 12)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 48, 18)]},
    ], duration=60.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert sum(seg["src_out"] - seg["src_in"] for seg in clip["segments"]) > 15


def test_mean_db_cache_preserves_angle_pick(tmp_path):
    # Behavior-parity guard for the RMS-caching refactor: build_plan must
    # decode each cam's audio once (via an internal rms_cache) rather than
    # re-decoding the whole file per peak per cam, but the angle pick must
    # be identical to the pre-refactor per-call rms_db() behavior. Uses
    # three sub-cams (not just camB/camC) with a clearly distinct loudest
    # one, so a cache bug that mixes up per-cam (times, db) arrays -- e.g.
    # reusing camA's or another cam's array for the wrong cam -- would pick
    # the wrong angle and fail this test.
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 6)]},
        {"name": "camC", "color": "blue", "offset": 0.0, "bursts": [(30, 34, 22)]},
        {"name": "camD", "color": "yellow", "offset": 0.0, "bursts": [(30, 34, 4)]},
    ], duration=50.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    # A cache bug that mixed up per-cam (times, db) arrays would put the wrong
    # camera on screen; deriving the expected one from the audio catches it.
    # Four cameras hearing one burst gives no second angle to switch to -- the
    # camera that heard it is already the one showing it -- so this checks the
    # camera, not the number of segments.
    start = min(g["src_in"] - offsets.get(g["cam"], 0.0) for g in clip["segments"])
    end = max(g["src_out"] - offsets.get(g["cam"], 0.0) for g in clip["segments"])
    assert clip["segments"][0]["cam"] == _by_loudness(res, offsets, start, end, cfg)[0]


def _cfg_yaml(tmp_path, reaction=None, **top):
    import yaml
    d = {"build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 20,
         "peak": {"min_gap_sec": 5, "threshold_k": 2.0}}
    d.update(top)
    if reaction:
        d["reaction"] = reaction
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(d))
    return load_config(str(p))


def _multicam_res(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 18)]},
    ], duration=50.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    offsets = compute_offsets(res["audio"], "camA")
    return res, offsets


def _reaction_res(tmp_path):
    """A goal one camera hears and a celebration the other hears.

    Switching angle is not automatic any more -- the second angle has to be the
    one hearing the celebration, or the clip stays on the camera it started on.
    So a fixture that exercises the cut needs the two moments genuinely split:
    camA carries the goal, camB takes over as the crowd keeps going.
    """
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(28, 35, 20)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(32, 36, 20)]},
    ], duration=55.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    return res, compute_offsets(res["audio"], "camA")


def test_angle_cut_holds_post_goal(tmp_path):
    # The Cam-A buildup must hold PAST the goal peak T before switching to
    # the reaction cam (not cut exactly at T), and the reaction segment must
    # start where the buildup ends and survive with >= min_reaction_sec.
    res, offsets = _reaction_res(tmp_path)
    cfg = _cfg_yaml(tmp_path, reaction={"post_goal_sec": 2.5, "min_reaction_sec": 2})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    T = clip["T"]
    a, b = clip["segments"]
    assert len(clip["segments"]) == 2
    assert a["src_out"] > T                                   # holds past the goal
    # src_out is on the primary camera's clock; take the offset back out before
    # comparing against T, which lives on the reference clock.
    cut_real = a["src_out"] - offsets.get(a["cam"], 0.0)
    assert cut_real <= T + cfg.post_goal_sec + 0.05           # never more than post_goal
    # both sides of the cut, back on the reference clock: no gap, no overlap
    assert abs((b["src_in"] - offsets.get(b["cam"], 0.0))
               - (a["src_out"] - offsets.get(a["cam"], 0.0))) < 0.1
    assert (b["src_out"] - b["src_in"]) >= cfg.min_reaction_sec - 0.6


def test_post_goal_clamped_to_end(tmp_path):
    # With a post_goal larger than the whole celebration, the cut must clamp
    # to end - min_reaction (not run past the clip), keeping the reaction
    # segment exactly min_reaction_sec long.
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg_yaml(tmp_path, reaction={"post_goal_sec": 100, "min_reaction_sec": 2})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    a, b = plan["clips"][0]["segments"]
    end = b["src_out"]                                        # offset ~0
    assert abs(a["src_out"] - (end - cfg.min_reaction_sec)) < 0.1   # cut clamped to end-min_reaction
    assert abs((b["src_out"] - b["src_in"]) - cfg.min_reaction_sec) < 0.1


def test_short_reaction_cut_is_suppressed(tmp_path):
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg_yaml(tmp_path, reaction={"post_goal_sec": 7.0, "min_reaction_sec": 2.0,
                                        "min_angle_switch_sec": 3.0})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert len(clip["segments"]) == 1


def _reaction_cfg():
    return Config(build_up_sec=5, min_len_sec=8, max_len_sec=25,
                  margin_db=6, hold_sec=2, rms_window_sec=0.5)


def test_cheer_onset_precedes_peak():
    # Quiet ambient (-40dB), then the cheer RISES at t=20 (onset) and only reaches
    # its loudness PEAK at t~25. The clip anchor must land on the onset (~goal
    # moment), well before the loudness peak, so the shot isn't cut off the front.
    cfg = _reaction_cfg()                        # build_up 5, margin_db 6
    times = np.arange(80) * 0.5                  # 0 .. 39.5
    db = np.full(times.shape, -40.0)
    db[(times >= 20) & (times < 28)] = -12.0     # cheer: loud from the onset at t=20
    db[(times >= 24) & (times < 26)] = -8.0      # loudness peak a few seconds later
    onset = cheer_onset(times, db, 24.5, cfg)    # peak sits at ~24.5
    assert 19.5 <= onset <= 20.5                  # anchored on the rising edge (goal moment)
    assert onset < 24.5 - 2                        # comfortably before the loudness peak


def test_build_plan_anchors_on_onset(tmp_path):
    # End-to-end: the clip's start is build_up before the ONSET, and the plan
    # records both the onset (T) and the loudness peak.
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg(tmp_path)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    assert "peak" in clip and clip["T"] <= clip["peak"]      # onset at or before peak
    # The clip must start build_up before the anchor in REAL time. src_in is on
    # the primary camera's own clock, so the offset has to come back out first --
    # asserting the raw number only held while the primary was always camA.
    g = clip["segments"][0]
    real_in = g["src_in"] - offsets.get(g["cam"], 0.0)
    assert abs(real_in - max(0.0, clip["T"] - cfg.build_up_sec)) < 0.05


def test_build_plan_records_which_net_was_hit(tmp_path, monkeypatch):
    """`_refine_anchor`'s verdict reaches the clip, and moves the anchor with it.

    It used to pick the angle too. It no longer does -- see
    `test_angle_comes_from_sound_not_the_net_roi` -- but which net the ball went
    into is still worth knowing, and the UI shows it.
    """
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)                      # camA + camB
    (tmp_path / "net_rois.json").write_text('{"cam":[0,0,1,1]}')  # non-empty -> rois truthy
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "scan_enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})
    monkeypatch.setattr(sp, "_refine_anchor", lambda *a, **k: (30.0, "camB"))
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] == "camB"
    assert clip["T"] == 30.0


def test_build_plan_prefers_earlier_roi_hit_over_stronger_later_hit(tmp_path, monkeypatch):
    # A wider angle can produce a stronger ROI motion score after the real goal.
    # The planner should still pick the earliest valid net hit so the goal-side
    # camera starts with buildup instead of cutting in after the ball is already in.
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "scan_enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})

    def fake_locate(source, center_time, cfg, roi):
        if source.endswith("camA.mp4"):
            return {"goal_time": 31.0, "confidence": 40.0}
        return {"goal_time": 29.5, "confidence": 8.0}

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", fake_locate)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] == "camB"
    assert clip["segments"][0]["cam"] == "camB"


def test_build_plan_ignores_roi_hit_that_is_too_far_before_onset(tmp_path, monkeypatch):
    # Widening the ROI window helps delayed crowd reactions, but an old net bump
    # should not steal the clip from a plausible goal-side hit near this cheer.
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "scan_enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})

    def fake_locate(source, center_time, cfg, roi):
        if source.endswith("camA.mp4"):
            return {"goal_time": center_time - 12.0, "confidence": 80.0}
        return {"goal_time": center_time - 2.0, "confidence": 8.0}

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", fake_locate)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] == "camB"
    assert clip["segments"][0]["cam"] == "camB"


def test_build_plan_prefers_near_onset_roi_over_stale_early_hit(tmp_path, monkeypatch):
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})

    def fake_locate(source, center_time, cfg, roi):
        if source.endswith("camA.mp4"):
            return {"goal_time": center_time + 0.5, "confidence": 9.0}
        return {"goal_time": center_time - 9.0, "confidence": 16.0}

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", fake_locate)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] == "camA"
    # The point is which hit the anchor took, not which camera ends up on screen
    # -- the angle comes from loudness now. The stale hit sits 9s back, so a
    # small peak-to-anchor gap means the near one won.
    assert clip["peak"] - clip["T"] < 5.0


def test_build_plan_prefers_near_pre_onset_roi_over_stale_early_hit(tmp_path, monkeypatch):
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})

    def fake_locate(source, center_time, cfg, roi):
        if source.endswith("camA.mp4"):
            return {"goal_time": center_time - 0.35, "confidence": 8.0}
        return {"goal_time": center_time - 8.75, "confidence": 11.0}

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", fake_locate)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] == "camA"
    assert clip["peak"] - clip["T"] < 5.0      # took the near hit, not the 8.75s one


def test_build_plan_keeps_audio_onset_when_only_roi_hit_is_too_early(tmp_path, monkeypatch):
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "scan_enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal",
                        lambda source, center_time, cfg, roi: {"goal_time": center_time - 12.0,
                                                               "confidence": 80.0})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] is None
    assert clip["segments"][0]["cam"] == "camB"   # no ROI answer -> loudest cam


def test_build_plan_adds_roi_only_clip_when_audio_has_no_peak(tmp_path, monkeypatch):
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "scan_enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})

    def fake_scan(source, cfg, roi, min_gap_sec, **kw):
        if source.endswith("camB.mp4"):
            return [{"goal_time": 30.0, "confidence": 20.0}]
        return []

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "scan_goal_events", fake_scan)
    plan = sp.build_plan(res, offsets, [], "camA", cfg)
    assert len(plan["clips"]) == 1
    clip = plan["clips"][0]
    assert clip["goal_cam"] == "camB"
    assert clip["segments"][0]["cam"] == "camB"
    assert clip["roi_only"] is True          # flagged for review/verification
    assert clip["scan_conf"] == 20.0         # net-motion prominence carried for review sorting


def test_build_plan_adds_weak_subcam_audio_roi_candidate(tmp_path, monkeypatch):
    import src.segment_planner as sp

    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json"),
                                      "weak_audio_k": 2.0,
                                      "weak_audio_min_confidence": 40.0})
    times = np.arange(0.0, 140.0, 0.5)

    def fake_rms(path, _window):
        db = np.full_like(times, -40.0)
        # Both cams have a weak local audio rise. Only the non-anchor cam should
        # be allowed to create a new ROI-backed candidate.
        db[60 if path == "camA.wav" else 62] = -28.0
        return times, db

    calls = []

    def fake_locate(source, center_time, cfg, roi):
        calls.append(source)
        return {"goal_time": center_time - 1.5, "confidence": 80.0}

    monkeypatch.setattr(sp, "rms_db", fake_rms)
    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", fake_locate)
    pre = {"cams": ["camA", "camB"], "is_multicam": True,
           "audio": {"camA": "camA.wav", "camB": "camB.wav"},
           "source": {"camA": "camA.mp4", "camB": "camB.mp4"},
           "width": 1920, "height": 1080}
    plan = sp.build_plan(pre, {"camA": 0.0, "camB": 0.0}, [], "camA", cfg)

    assert calls == ["camB.mp4"]
    assert len(plan["clips"]) == 1
    assert plan["clips"][0]["goal_cam"] == "camB"
    assert plan["clips"][0]["roi_only"] is True


def test_build_plan_keeps_weak_subcam_roi_as_separate_close_clip(tmp_path, monkeypatch):
    import src.segment_planner as sp

    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json"),
                                      "weak_audio_k": 2.0,
                                      "weak_audio_min_confidence": 40.0})
    times = np.arange(0.0, 140.0, 0.5)

    def fake_rms(path, _window):
        db = np.full_like(times, -40.0)
        if path == "camA.wav":
            db[218] = -20.0  # strong main-cam anchor at 109.0
        else:
            db[231] = -28.0  # weaker sub-cam candidate at 115.5
        return times, db

    def fake_locate(source, center_time, cfg, roi):
        if source == "camB.mp4":
            return {"goal_time": center_time - 0.5, "confidence": 120.0}
        return None

    monkeypatch.setattr(sp, "rms_db", fake_rms)
    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", fake_locate)
    pre = {"cams": ["camA", "camB"], "is_multicam": True,
           "audio": {"camA": "camA.wav", "camB": "camB.wav"},
           "source": {"camA": "camA.mp4", "camB": "camB.mp4"},
           "width": 1920, "height": 1080}
    plan = sp.build_plan(pre, {"camA": 0.0, "camB": 0.0}, [109.0], "camA", cfg)

    assert len(plan["clips"]) == 2
    assert plan["clips"][0]["roi_only"] is False
    assert plan["clips"][1]["roi_only"] is True
    assert plan["clips"][1]["goal_cam"] == "camB"


def test_audio_backed_clips_not_flagged_roi_only(tmp_path):
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg(tmp_path)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    assert all(c["roi_only"] is False for c in plan["clips"])
    assert all(c["scan_conf"] is None for c in plan["clips"])   # audio clips have no scan prominence


def test_build_plan_does_not_duplicate_roi_only_near_audio_peak(tmp_path, monkeypatch):
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "scan_enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", lambda *a, **k: None)
    monkeypatch.setattr(sp.goal_locator, "scan_goal_events",
                        lambda *a, **k: [{"goal_time": 31.0, "confidence": 20.0}])
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = sp.build_plan(res, offsets, peaks, "camA", cfg)
    assert len(plan["clips"]) == 1


def test_no_locate_falls_back_to_whoever_heard_it(tmp_path):
    # Without net-ROI there is no answer to "which goal", so the buildup used to
    # default to camA -- which shows the far end of the pitch from across it half
    # the time. The camera that heard the moment loudest is the one nearest it.
    # On the game this was measured against the fallback agreed with the ROI's
    # answer 70% of the time, against 26% for always-camA.
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg(tmp_path)                                        # locate disabled
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] is None
    assert clip["segments"][0]["cam"] == "camB"   # the louder cam in this fixture


def test_adjacent_clips_do_not_overlap(tmp_path):
    # Two goals 12s apart with LONG celebrations: without clamping, clip 1's end
    # (up to start+max_len=20s) would run past clip 2's start and duplicate the
    # same scene in both clips. The planner must clamp clip 1 to clip 2's start.
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         # two long cheers with a clear 2s dip between (distinct onsets ~20 / ~32)
         "bursts": [(20, 30, 12), (32, 42, 12)]},
    ], duration=60.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)                             # min_gap 5, build_up 5, max_len 20
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    assert len(peaks) >= 2
    plan = build_plan(res, {"camA": 0.0}, peaks, "camA", cfg)
    clips = plan["clips"]
    for a, b in zip(clips, clips[1:]):
        end_a = a["segments"][-1]["src_out"]         # single-cam: camA timeline
        start_b = b["segments"][0]["src_in"]
        assert end_a <= start_b + 1e-6, f"clip overlap: {end_a} > {start_b}"


def test_reaction_end_extends_for_long_celebration():
    # Quiet ambient (~-40dB) everywhere except a long, loud celebration
    # (~-10dB) from T=10 through t=30 (20s), followed by a quiet tail.
    # Pre-T baseline (median over [5,10]) is -40dB, so quiet_level is
    # -34dB and the loud celebration never looks "settled" until it
    # actually quiets back down near t=30 -> the clip should extend well
    # past min_len, clamped at hi = start(5) + max_len(25) = 30.0.
    cfg = _reaction_cfg()
    T = 10.0
    times = np.arange(72) * 0.5   # 0.0 .. 35.5s
    db = np.full(times.shape, -40.0)
    db[(times >= 10) & (times < 30)] = -10.0

    end = reaction_end(times, db, T, cfg)

    start = max(0.0, T - cfg.build_up_sec)
    assert end >= start + cfg.min_len_sec + 3  # i.e. > 16.0: genuinely extended
    assert end == 30.0                          # clamped at hi, celebration ran to hi


def test_reaction_end_short_for_brief_celebration():
    # Same quiet ambient, but the celebration is brief (T=10 to t=12,
    # 2s) then back to ambient -> should settle quickly and clamp near
    # min_len, not get dragged out.
    cfg = _reaction_cfg()
    T = 10.0
    times = np.arange(72) * 0.5   # 0.0 .. 35.5s
    db = np.full(times.shape, -40.0)
    db[(times >= 10) & (times < 12)] = -10.0

    end = reaction_end(times, db, T, cfg)

    start = max(0.0, T - cfg.build_up_sec)
    assert end <= start + cfg.min_len_sec + 3  # i.e. <= 16.0: stays near min_len


def test_main_cam_beats_the_loudness_guess(tmp_path):
    """`main_cam` is documented as the buildup reference -- the user naming a
    camera is an instruction, and inferring a different one from loudness would
    quietly ignore it."""
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 4)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 20)]},
    ], duration=50.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    offsets = compute_offsets(res["audio"], "camA")

    quiet_but_chosen = _cfg_yaml(tmp_path, main_cam="camA")
    peaks = detect_peaks(res["audio"]["camA"], quiet_but_chosen)
    clip = build_plan(res, offsets, peaks, "camA", quiet_but_chosen)["clips"][0]
    assert clip["segments"][0]["cam"] == "camA"

    # ...and with nothing chosen, the louder camera does take it.
    unset = _cfg_yaml(tmp_path)
    clip = build_plan(res, offsets, peaks, "camA", unset)["clips"][0]
    assert clip["segments"][0]["cam"] == "camB"


def test_clips_do_not_overlap_when_anchors_land_close(tmp_path):
    """Anchors are not evenly spaced by the time clips are built: `_refine_anchor`
    moves one to the net-motion spike up to 11s ahead of its cheer, so two that
    cleared min_gap as peaks can end up seconds apart. Only the clip END was
    clamped, and its floor wins in exactly that case -- so the next clip's
    buildup reached back across it and both showed the same footage. Measured on
    one real game: 12 overlapping pairs, the worst sharing 8 of its 10 seconds.

    Reproduced with a small min_gap against a larger build_up, which puts anchors
    closer together than one buildup without needing ROI calibration. Compared in
    reference time -- the single-cam test above compares raw src times, which
    only holds while every segment is on one clock.
    """
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(20, 22, 12), (26, 28, 12), (32, 34, 12)]},
        {"name": "camB", "color": "green", "offset": 0.0,
         "bursts": [(20, 22, 16), (26, 28, 8), (32, 34, 16)]},
    ], duration=60.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg_yaml(tmp_path, peak={"min_gap_sec": 3, "threshold_k": 2.0})
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    assert len(peaks) >= 2, "인접한 피크가 두 개 이상이어야 재현된다"
    clips = build_plan(res, offsets, peaks, "camA", cfg)["clips"]

    def span(c):
        lo = min(g["src_in"] - offsets.get(g["cam"], 0.0) for g in c["segments"])
        hi = max(g["src_out"] - offsets.get(g["cam"], 0.0) for g in c["segments"])
        return lo, hi

    for a, b in zip(clips, clips[1:]):
        assert span(a)[1] <= span(b)[0] + 0.05, \
            f"clips overlap in real time: {span(a)} then {span(b)}"


def test_distant_net_hit_is_not_called_this_goal(tmp_path, monkeypatch):
    """A net hit far before the cheer belongs to the previous phase of play.
    Taking it drags the clip off the goal and the angle with it -- three angles
    confirmed wrong by eye on one game all came from hits 9.4s or more before
    their cheer, against a 3.8s median for the ones that were right.
    """
    from src import segment_planner as SP
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "rois_path": "x.json",
                                      "max_lead_sec": 9.0})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    onset_seen = {}

    def fake_locate(source, center_time, c, roi):
        # One camera reports a hit far earlier than the cheer; nothing else does.
        onset_seen.setdefault("t", center_time)
        if source.endswith(res["source"]["camB"].split("/")[-1]):
            return {"goal_time": center_time - 10.0, "confidence": 99.0}
        return None

    monkeypatch.setattr(SP.goal_locator, "load_rois", lambda p: {"camA": [0, 0, 1, 1],
                                                                 "camB": [0, 0, 1, 1]})
    monkeypatch.setattr(SP.goal_locator, "roi_for_cam", lambda cam, r, s=None: [0, 0, 1, 1])
    monkeypatch.setattr(SP.goal_locator, "locate_goal", fake_locate)
    clip = build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] is None, "10s 앞선 히트는 이 골로 인정하면 안 된다"

    loose = _cfg_yaml(tmp_path, locate={"enabled": True, "rois_path": "x.json",
                                        "max_lead_sec": 12.0})
    clip = build_plan(res, offsets, peaks, "camA", loose)["clips"][0]
    assert clip["goal_cam"] == "camB", "창을 넓히면 같은 히트를 받아들여야 한다"


def test_angle_comes_from_sound_not_the_net_roi(tmp_path, monkeypatch):
    """Each camera sits beside one goal, so its own net fills the frame and the
    keeper standing in it moves more pixels than a ball entering the far net.
    On five clips checked by eye the near-net camera won on ROI motion every
    time and was wrong every time, while loudness named the right camera in all
    five. So the ROI says when, and sound says where to look.
    """
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)
    (tmp_path / "net_rois.json").write_text('{"any":[0,0,1,1]}')
    locate = {"enabled": True, "rois_path": str(tmp_path / "net_rois.json")}

    def fake_locate(source, center_time, c, roi):
        # camA's net "fires"; camB is the one that actually heard the moment.
        if source.endswith("camA.mp4"):
            return {"goal_time": center_time, "confidence": 30.0}
        return None

    monkeypatch.setattr(sp.goal_locator, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])
    monkeypatch.setattr(sp.goal_locator, "locate_goal", fake_locate)

    cfg = _cfg_yaml(tmp_path, locate=locate)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] == "camA", "어느 네트였는지는 계속 기록되어야 한다"
    assert clip["segments"][0]["cam"] == "camB", "앵글은 더 크게 들은 카메라"

    # The toggle puts the old behaviour back for a tightly-drawn ROI.
    cfg = _cfg_yaml(tmp_path, locate={**locate, "angle_from_roi": True})
    clip = sp.build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["segments"][0]["cam"] == "camA"


def test_the_angle_vote_hears_the_whole_clip_not_just_after_the_anchor():
    """A camera whose cheer lands in the buildup must still win.

    The anchor moves: the net-ROI drags it to the goal frame, which can be
    seconds ahead of the crowd. A vote window starting at the anchor therefore
    moves with the ROI, and on the game measured here it walked far enough
    forward at 10:28 that the cheer which proposed the clip fell outside it --
    handing the angle back to the ROI through the back door, and to whichever
    camera had the higher noise floor.
    """
    import numpy as np
    from src.segment_planner import _loudest_cam
    t = np.arange(0.0, 40.0, 0.5)
    near = np.where((t >= 22) & (t <= 25), 3.0, 0.0)   # one burst, in the buildup
    far = np.full_like(t, 0.4)                          # a steadily noisy floor
    k = {"near": (t, near), "far": (t, far)}
    cams, offs = ["near", "far"], {}

    assert _loudest_cam(cams, offs, 20.0, 35.0, k) == "near"   # whole clip
    assert _loudest_cam(cams, offs, 28.0, 35.0, k) == "far"    # anchor onward: lost


def test_clip_cannot_end_before_its_own_cheer():
    """The quiet gap between a ROI-moved anchor and the crowd is not the end.

    The net-ROI pulls the anchor back to the goal frame, which can be many
    seconds ahead of the cheer, and everything in between is quiet by
    definition. Searching for "settled" from the anchor found that gap and
    closed the clip inside it -- so a clip proposed by a cheer could finish
    before the cheer, which is to say before the goal. It was not rare: on
    three real games, 24%, 24% and 26% of clips left under a second after
    their own peak, the worst ending 6.4s before it.
    """
    import numpy as np
    from src.segment_planner import reaction_end

    cfg = Config()
    cfg.build_up_sec, cfg.min_len_sec, cfg.max_len_sec = 5.0, 10.0, 25.0
    cfg.hold_sec, cfg.margin_db = 1.0, 3.0
    times = np.arange(0.0, 80.0, 0.5)
    T, peak = 30.0, 39.0                     # ROI put the anchor 9s early
    db = np.where((times >= peak) & (times <= peak + 6), -20.0, -50.0)

    assert reaction_end(times, db, T, cfg, peak) > peak, "함성 전에 끝나면 안 된다"
    # Without the peak it settles in the gap, at the min-length floor.
    assert reaction_end(times, db, T, cfg) < peak
