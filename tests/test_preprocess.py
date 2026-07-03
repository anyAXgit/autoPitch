import os
import subprocess
import pytest
import soundfile as sf
from tests.make_dummy import make_dummy_set
from src.preprocess import list_raw, cam_id, preprocess_all


def _frame_rates(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,avg_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True).stdout.split()
    return out  # [r_frame_rate, avg_frame_rate], e.g. ["30/1", "30/1"]


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


def test_empty_raw_dir_raises(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    tv, ta = tmp_path / "tv", tmp_path / "ta"
    with pytest.raises(FileNotFoundError):
        preprocess_all(str(raw), str(tv), str(ta), fps=30)


def test_cfr_output_frame_rate(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(2, 4, 10)]},
    ], duration=8.0)
    tv, ta = tmp_path / "tv", tmp_path / "ta"
    res = preprocess_all(str(raw), str(tv), str(ta), fps=30)
    r_rate, avg_rate = _frame_rates(res["video"]["camA"])

    def _to_float(fraction):
        num, den = fraction.split("/")
        return float(num) / float(den)

    assert _to_float(r_rate) == 30.0
    assert _to_float(avg_rate) == 30.0


def test_multicam_flag_true(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(2, 4, 10)]},
        {"name": "camB", "color": "green", "offset": 0.5, "bursts": [(2.5, 4.5, 10)]},
    ], duration=8.0)
    tv, ta = tmp_path / "tv", tmp_path / "ta"
    res = preprocess_all(str(raw), str(tv), str(ta), fps=30)
    assert res["is_multicam"] is True
    assert res["cams"] == ["camA", "camB"]
