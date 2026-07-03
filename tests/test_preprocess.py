import os
import soundfile as sf
from tests.make_dummy import make_dummy_set
from src.preprocess import list_raw, cam_id, preprocess_all


def test_cam_ordering_and_multicam_flag(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(10, 12, 10)]},
        {"name": "camB", "color": "green", "offset": 1.5, "bursts": [(11.5, 13.5, 10)]},
    ], duration=20.0)
    files = list_raw(str(raw))
    assert [cam_id(f) for f in files] == ["camA", "camB"]


def test_preprocess_outputs(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(10, 12, 10)]},
    ], duration=20.0)
    tv, ta = tmp_path / "tv", tmp_path / "ta"
    res = preprocess_all(str(raw), str(tv), str(ta), fps=30)
    assert res["cams"] == ["camA"]
    assert res["is_multicam"] is False
    assert os.path.exists(res["video"]["camA"])
    wav = res["audio"]["camA"]
    data, sr = sf.read(wav)
    assert sr == 16000
    assert data.ndim == 1        # mono
