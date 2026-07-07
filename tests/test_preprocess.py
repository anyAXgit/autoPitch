import os
import pytest
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
    # source keeps the ORIGINAL raw path (no video re-encode in preprocess)
    assert os.path.exists(res["source"]["camA"])
    assert cam_id(res["source"]["camA"]) == "camA"
    wav = res["audio"]["camA"]
    data, sr = sf.read(wav)
    assert sr == 16000
    assert data.ndim == 1        # mono


def test_preprocess_extracts_audio_no_video(tmp_path):
    # V1: preprocess extracts only the analysis WAV + a source map pointing at
    # the originals. It must NOT write any normalized video to temp_video
    # (normalization moved to render_segment).
    import glob
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(10, 12, 10)]},
    ], duration=20.0)
    tv, ta = tmp_path / "tv", tmp_path / "ta"
    res = preprocess_all(str(raw), str(tv), str(ta), fps=30)
    assert os.path.exists(res["audio"]["camA"])
    data, sr = sf.read(res["audio"]["camA"])
    assert sr == 16000 and data.ndim == 1
    assert os.path.exists(res["source"]["camA"])
    assert res["width"] == 1920 and res["height"] == 1080 and res["fps"] == 30
    # no transcoded video written to temp_video
    assert glob.glob(str(tv / "*")) == []


def test_empty_raw_dir_raises(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    tv, ta = tmp_path / "tv", tmp_path / "ta"
    with pytest.raises(FileNotFoundError):
        preprocess_all(str(raw), str(tv), str(ta), fps=30)


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


# Cross-resolution normalization is now a RENDER-time property (render_segment
# scale+pad), covered by tests/test_video_editor.py::test_render_multicam_mixed_resolution.
# CFR output frame rate likewise: render_segment emits -vsync cfr -r fps.
