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
    assert len(clip["segments"]) == 2
    assert clip["segments"][0]["cam"] == "camA"            # build-up
    assert clip["segments"][1]["cam"] == "camB"            # louder reaction


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
    assert len(clip["segments"]) == 2
    assert clip["segments"][0]["cam"] == "camA"             # build-up
    assert clip["segments"][1]["cam"] == "camC"             # loudest reaction (gain 22)


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


def test_angle_cut_holds_post_goal(tmp_path):
    # The Cam-A buildup must hold PAST the goal peak T before switching to
    # the reaction cam (not cut exactly at T), and the reaction segment must
    # start where the buildup ends and survive with >= min_reaction_sec.
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg_yaml(tmp_path, reaction={"post_goal_sec": 2.5, "min_reaction_sec": 2})
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    T = clip["T"]
    a, b = clip["segments"]
    assert len(clip["segments"]) == 2
    assert a["src_out"] > T                                   # holds past the goal
    assert a["src_out"] <= T + cfg.post_goal_sec + 1e-6       # never more than post_goal
    assert abs(b["src_in"] - a["src_out"]) < 0.6             # reaction starts at the cut (offset~0)
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
    assert abs(clip["segments"][0]["src_in"] - max(0.0, clip["T"] - cfg.build_up_sec)) < 1e-6


def test_build_plan_goal_side_cam_is_primary(tmp_path, monkeypatch):
    # When net-ROI identifies the goal-side cam (here camB), the GOAL segment
    # (seg0) must use camB, not always camA -- so the goal is shown from the
    # camera nearest to where it was scored; reaction = the other cam (camA).
    import src.segment_planner as sp
    res, offsets = _multicam_res(tmp_path)                      # camA + camB
    (tmp_path / "net_rois.json").write_text('{"cam":[0,0,1,1]}')  # non-empty -> rois truthy
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True, "scan_enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})
    monkeypatch.setattr(sp, "_refine_anchor", lambda *a, **k: (30.0, "camB"))
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = sp.build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    assert clip["goal_cam"] == "camB"
    assert clip["segments"][0]["cam"] == "camB"                 # goal from goal-side cam
    assert clip["segments"][1]["cam"] == "camA"                 # reaction from the other cam


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
    assert clip["segments"][0]["cam"] == "camA"


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
    assert clip["segments"][0]["cam"] == "camA"


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
    assert clip["segments"][0]["cam"] == "camA"


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


def test_build_plan_no_locate_keeps_camA_primary(tmp_path):
    # Without net-ROI, behavior is unchanged: Cam A stays the buildup/goal angle.
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg(tmp_path)                                        # locate disabled
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] is None
    assert clip["segments"][0]["cam"] == "camA"


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
