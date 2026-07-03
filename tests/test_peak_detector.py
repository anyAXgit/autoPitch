from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.config import load_config
from src.peak_detector import detect_peaks


def _cfg(tmp_path, **over):
    import yaml
    base = {"peak": {"min_gap_sec": 5, "threshold_k": 2.0}}
    base["peak"].update(over)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base))
    return load_config(str(p))


def test_detects_two_bursts(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(10.0, 12.0, 12.0), (30.0, 32.0, 12.0)]},
    ], duration=45.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    peaks = detect_peaks(res["audio"]["camA"], _cfg(tmp_path))
    assert len(peaks) == 2
    assert abs(peaks[0] - 10.0) < 1.5
    assert abs(peaks[1] - 30.0) < 1.5


def test_max_clips_caps_and_sorts(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(10, 12, 8.0), (30, 32, 20.0), (50, 52, 12.0)]},
    ], duration=65.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    peaks = detect_peaks(res["audio"]["camA"], _cfg(tmp_path, max_clips=2))
    assert len(peaks) == 2
    # loudest are the 30s (gain 20) and 50s (gain 12); result sorted ascending
    assert abs(peaks[0] - 30.0) < 1.5
    assert abs(peaks[1] - 50.0) < 1.5
