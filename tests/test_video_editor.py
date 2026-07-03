import os
import subprocess
from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.config import load_config
from src.sync_engine import compute_offsets
from src.peak_detector import detect_peaks
from src.segment_planner import build_plan
from src.video_editor import render_plan, probe_duration


def _cfg(tmp_path):
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 18,
        "crossfade_sec": 0.5, "peak": {"min_gap_sec": 5, "threshold_k": 2.0},
    }))
    return load_config(str(p))


def test_render_multicam_clip(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 18)]},
    ], duration=55.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    cfg = _cfg(tmp_path)
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    out_dir = tmp_path / "out"
    paths = render_plan(plan, str(out_dir))
    assert len(paths) == 1
    assert os.path.exists(paths[0])
    assert os.path.exists(os.path.join(str(out_dir), "highlight_all.mp4"))
    # clip has audio + video streams
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", paths[0]], capture_output=True, text=True, check=True
    ).stdout
    assert "video" in streams and "audio" in streams
    # dynamic length is within clamp bounds
    # 2-segment clips render crossfade_sec shorter than the planned window (xfade overlaps content)
    assert 7.0 <= probe_duration(paths[0]) <= 19
