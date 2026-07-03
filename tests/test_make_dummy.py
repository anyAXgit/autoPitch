import os
import subprocess
from tests.make_dummy import make_dummy_set


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def test_make_dummy_set_creates_files(tmp_path):
    cams = [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(10.0, 12.0, 10.0), (30.0, 32.0, 10.0)]},
        {"name": "camB", "color": "green", "offset": 1.5,
         "bursts": [(11.5, 13.5, 10.0), (31.5, 33.5, 14.0)]},
    ]
    gt = make_dummy_set(str(tmp_path), cams, duration=45.0, fps=30)
    assert set(gt["files"]) == {"camA", "camB"}
    for name, path in gt["files"].items():
        assert os.path.exists(path)
        assert abs(_duration(path) - 45.0) < 0.5
