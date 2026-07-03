from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.config import load_config
from src.segment_planner import build_plan
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
