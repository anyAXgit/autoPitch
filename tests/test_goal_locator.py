import json
import subprocess

import numpy as np

from src.config import Config, LocateConfig
from src import goal_locator
from src.goal_locator import load_rois, roi_for_cam, locate_goal
from src.segment_planner import _refine_anchor

# ROI over a box drawn at x=200,y=40,w=70,h=70 in a 320x240 frame (normalized)
_ROI = [200 / 320, 40 / 240, 70 / 320, 70 / 240]
_CFG = Config(locate=LocateConfig(enabled=True, pre_sec=2.0, post_sec=3.0,
                                  fps=15.0, min_prominence=6.0, frame_px=64))


def _flash_clip(path, flash=True):
    # black clip; optionally flash a white box in the ROI during [3.0, 3.3]s
    vf = ("drawbox=x=200:y=40:w=70:h=70:color=white:t=fill:"
          "enable='between(t,3,3.3)'") if flash else "null"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=320x240:d=6:r=15",
         "-vf", vf, "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def test_locate_goal_finds_net_spike(tmp_path):
    clip = tmp_path / "flash.mp4"
    _flash_clip(clip, flash=True)
    r = locate_goal(str(clip), 3.15, _CFG, _ROI)      # onset ~ near the flash
    assert r is not None
    assert 2.7 <= r["goal_time"] <= 3.5               # spike localized to the net event
    assert r["confidence"] >= _CFG.locate.min_prominence


def test_locate_goal_none_without_spike(tmp_path):
    clip = tmp_path / "static.mp4"
    _flash_clip(clip, flash=False)                    # no net disturbance
    assert locate_goal(str(clip), 3.15, _CFG, _ROI) is None   # -> caller falls back to onset


def test_locate_goal_anchors_to_motion_run_start(monkeypatch):
    cfg = Config(locate=LocateConfig(enabled=True, pre_sec=1.0, post_sec=1.0,
                                     fps=10.0, min_prominence=6.0, frame_px=1))
    # Frame diffs become [0, 0, 20, 20, 80, 0, ...]. The max is later in the
    # same net shake, but the goal anchor should be the first prominent frame.
    values = [0, 0, 0, 20, 40, 120, 120, 120, 120, 120]
    frames = np.array(values, dtype=np.float32).reshape(len(values), 1, 1)
    monkeypatch.setattr(goal_locator, "_roi_gray_frames", lambda *a, **k: frames)

    r = locate_goal("dummy.mp4", 10.0, cfg, [0, 0, 1, 1])

    assert r is not None
    assert abs(r["goal_time"] - 9.25) < 1e-6


def test_refine_anchor_uses_net_spike(tmp_path):
    clip = tmp_path / "flash.mp4"
    _flash_clip(clip, flash=True)
    pre = {"cams": ["camA"], "source": {"camA": str(clip)}}
    # ROI present -> (refined time ~3.0s, goal-side cam)
    got = _refine_anchor(pre, {"camA": 0.0}, 3.1, _CFG, {"flash.mp4": _ROI})
    assert got is not None and 2.7 <= got[0] <= 3.5 and got[1] == "camA"
    # no ROI matches this cam -> None (caller keeps the onset)
    assert _refine_anchor(pre, {"camA": 0.0}, 3.1, _CFG, {"other": _ROI}) is None


def test_roi_for_cam_substring_match():
    rois = {"DJI": [0.1, 0.1, 0.2, 0.2], "IMG": [0.5, 0.5, 0.2, 0.2]}
    assert roi_for_cam("DJI_20260703220255_0064_D", rois) == [0.1, 0.1, 0.2, 0.2]
    assert roi_for_cam("IMG_8414", rois) == [0.5, 0.5, 0.2, 0.2]
    assert roi_for_cam("cam1", rois, "/x/data/raw/cam1/DJI_20260703220255_0064_D.MP4") == [0.1, 0.1, 0.2, 0.2]
    assert roi_for_cam("camX", rois) is None


def test_roi_for_cam_prefers_file_specific_and_ignores_broad_cam_key():
    rois = {
        "cam1": [0.0, 0.0, 0.1, 0.1],
        "data/raw/cam1/game_a.mp4": [0.2, 0.2, 0.3, 0.3],
        "game_b.mp4": [0.4, 0.4, 0.5, 0.5],
    }
    assert roi_for_cam("cam1", rois, "/project/data/raw/cam1/game_a.mp4") == [0.2, 0.2, 0.3, 0.3]
    assert roi_for_cam("cam1", rois, "/project/data/raw/cam1/game_b.mp4") == [0.4, 0.4, 0.5, 0.5]
    assert roi_for_cam("cam1", rois, "/project/data/raw/cam1/game_c.mp4") is None


def test_load_rois_missing(tmp_path):
    assert load_rois(None) == {}
    assert load_rois(str(tmp_path / "nope.json")) == {}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"DJI": [0, 0, 1, 1]}))
    assert load_rois(str(p)) == {"DJI": [0, 0, 1, 1]}


def _moving_box_clip(path):
    # sustained ROI motion: a box that jumps around inside the ROI for 4s
    # (a keeper leaning on / fiddling with the net -- NOT a ball impact)
    vf = ("drawbox=x='200+mod(floor(t*8)*13\\,50)':y='40+mod(floor(t*8)*7\\,50)'"
          ":w=20:h=20:color=white:t=fill:enable='between(t,1,5)'")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=320x240:d=6:r=15",
         "-vf", vf, "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def test_impulse_gate_passes_flash_rejects_sustained(tmp_path):
    from src.goal_locator import event_impulse_ok
    flash = tmp_path / "flash.mp4"
    _flash_clip(flash, flash=True)                       # 0.3s impulse = net hit
    assert event_impulse_ok(str(flash), 3.15, _CFG, _ROI) is True
    busy = tmp_path / "busy.mp4"
    _moving_box_clip(busy)                               # 4s sustained motion = junk
    assert event_impulse_ok(str(busy), 3.0, _CFG, _ROI) is False


def test_scan_cache_skips_redecode(tmp_path, monkeypatch):
    from src.goal_locator import scan_goal_events
    clip = tmp_path / "flash.mp4"
    _flash_clip(clip, flash=True)
    cache = str(tmp_path / "scan_cache.json")
    calls = {"n": 0}
    orig = goal_locator._roi_gray_keyframes
    monkeypatch.setattr(goal_locator, "_roi_gray_keyframes",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or orig(*a, **k))
    first = scan_goal_events(str(clip), _CFG, _ROI, 5.0, cache_path=cache)
    assert calls["n"] == 1
    second = scan_goal_events(str(clip), _CFG, _ROI, 5.0, cache_path=cache)
    assert calls["n"] == 1                                # served from cache
    assert second == first
