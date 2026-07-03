import numpy as np

from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.config import Config, load_config
from src.segment_planner import build_plan, reaction_end
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


def _reaction_cfg():
    return Config(build_up_sec=5, min_len_sec=8, max_len_sec=25,
                  margin_db=6, hold_sec=2, rms_window_sec=0.5)


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
