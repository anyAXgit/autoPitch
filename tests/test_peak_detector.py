from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.config import load_config
from src.peak_detector import detect_peaks, detect_peaks_multicam


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


# ---- listening to every camera ----
# A goal at the far end is loud beside it and faint across from it. Detecting on
# one camera therefore drops half the pitch: measured on one real game, a cheer
# at k=4.76 on cam2 read 1.53 on cam1 and was never proposed.

def test_multicam_hears_what_the_reference_camera_missed(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        # camB is where the far-end burst happens; camA barely registers it.
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(10.0, 12.0, 12.0)]},
        {"name": "camB", "color": "blue", "offset": 0.0,
         "bursts": [(10.0, 12.0, 12.0), (30.0, 32.0, 12.0)]},
    ], duration=45.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"),
                         str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)

    alone = detect_peaks(res["audio"]["camA"], cfg)
    both = detect_peaks_multicam(res["audio"], {"camA": 0.0, "camB": 0.0},
                                 "camA", cfg)
    assert not any(28 <= t <= 34 for t in alone), "camA 혼자서는 못 듣는 상황이어야 한다"
    assert any(28 <= t <= 34 for t in both), "두 카메라를 합치면 잡혀야 한다"


def test_multicam_reports_on_the_reference_clock(tmp_path):
    """The other camera's peak has to come back in reference time, or every
    downstream cut lands at the wrong moment by exactly the offset."""
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": []},
        {"name": "camB", "color": "blue", "offset": 0.0,
         "bursts": [(20.0, 22.0, 12.0)]},
    ], duration=45.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"),
                         str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)
    # camB's recording starts 4s before camA's, so its 20s burst is camA's 16s.
    peaks = detect_peaks_multicam(res["audio"], {"camA": 0.0, "camB": 4.0},
                                  "camA", cfg)
    assert peaks, "버스트를 찾지 못했다"
    assert any(14 <= t <= 18 for t in peaks), f"기준 시계로 환산되지 않음: {peaks}"
