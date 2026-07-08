import json
import subprocess

from src.config import Config, LocateConfig
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


def test_refine_anchor_uses_net_spike(tmp_path):
    clip = tmp_path / "flash.mp4"
    _flash_clip(clip, flash=True)
    pre = {"cams": ["camA"], "source": {"camA": str(clip)}}
    # ROI present -> anchor refined to the net spike (~3.0s), overriding the onset
    got = _refine_anchor(pre, {"camA": 0.0}, 3.1, _CFG, {"camA": _ROI})
    assert got is not None and 2.7 <= got <= 3.5
    # no ROI matches this cam -> None (caller keeps the onset)
    assert _refine_anchor(pre, {"camA": 0.0}, 3.1, _CFG, {"other": _ROI}) is None


def test_roi_for_cam_substring_match():
    rois = {"DJI": [0.1, 0.1, 0.2, 0.2], "IMG": [0.5, 0.5, 0.2, 0.2]}
    assert roi_for_cam("DJI_20260703220255_0064_D", rois) == [0.1, 0.1, 0.2, 0.2]
    assert roi_for_cam("IMG_8414", rois) == [0.5, 0.5, 0.2, 0.2]
    assert roi_for_cam("camX", rois) is None


def test_load_rois_missing(tmp_path):
    assert load_rois(None) == {}
    assert load_rois(str(tmp_path / "nope.json")) == {}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"DJI": [0, 0, 1, 1]}))
    assert load_rois(str(p)) == {"DJI": [0, 0, 1, 1]}
