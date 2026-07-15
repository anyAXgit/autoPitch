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
    cfg = _cfg_yaml(tmp_path, locate={"enabled": True,
                                      "rois_path": str(tmp_path / "net_rois.json")})
    monkeypatch.setattr(sp, "_refine_anchor", lambda *a, **k: (30.0, "camB"))
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = sp.build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    assert clip["goal_cam"] == "camB"
    assert clip["segments"][0]["cam"] == "camB"                 # goal from goal-side cam
    assert clip["segments"][1]["cam"] == "camA"                 # reaction from the other cam


def test_build_plan_no_locate_keeps_camA_primary(tmp_path):
    # Without net-ROI, behavior is unchanged: Cam A stays the buildup/goal angle.
    res, offsets = _multicam_res(tmp_path)
    cfg = _cfg(tmp_path)                                        # locate disabled
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    clip = build_plan(res, offsets, peaks, "camA", cfg)["clips"][0]
    assert clip["goal_cam"] is None
    assert clip["segments"][0]["cam"] == "camA"


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
