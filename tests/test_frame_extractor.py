import subprocess

from tests.make_dummy import make_dummy_set
from src.preprocess import list_raw
from src.config import Config, VisionConfig
from src.frame_extractor import extract_goal_frames


def _height(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of",
         "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def test_extract_goal_frames_count_and_size(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
    ], duration=50.0)
    src = list_raw(str(raw))[0]
    cfg = Config(vision=VisionConfig(pre_sec=3, post_sec=8, fps=2, frame_height=120))

    frames = extract_goal_frames(src, 30.0, cfg, str(tmp_path / "frames"))

    # ~ (pre+post)*fps frames, small boundary slack
    expected = int((cfg.vision.pre_sec + cfg.vision.post_sec) * cfg.vision.fps)
    assert abs(len(frames) - expected) <= 2, f"got {len(frames)}, expected ~{expected}"
    assert frames == sorted(frames)
    for f in frames[:3]:
        assert _height(f) == cfg.vision.frame_height


def test_extract_goal_frames_clamps_window_start(tmp_path):
    # A goal near t=0 must not request negative source time; extraction still
    # yields frames (from [0, post]) rather than erroring.
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(1, 3, 10)]},
    ], duration=20.0)
    src = list_raw(str(raw))[0]
    cfg = Config(vision=VisionConfig(pre_sec=3, post_sec=8, fps=2, frame_height=120))

    frames = extract_goal_frames(src, 1.0, cfg, str(tmp_path / "frames"))
    assert len(frames) >= 1
