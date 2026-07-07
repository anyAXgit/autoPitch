from src.config import Config, VisionConfig
from src.goal_confirmer import confirm_goals


def _drop_all(frames):
    return {"is_goal": False, "confidence": 0.9}


def _drop_early(frames):
    # decide from the T encoded in the frame filename (T<time>_NNNN.jpg)
    import os
    name = os.path.basename(frames[0])
    t = float(name[1:name.index("_")])
    return {"is_goal": t >= 30, "confidence": 0.8}


def test_confirm_goals_passthrough_when_disabled():
    cfg = Config(vision=VisionConfig(enabled=False))
    peaks = [10.0, 20.0, 30.0]
    # even a drop-everything classifier is ignored while vision is off
    assert confirm_goals(peaks, {}, cfg, _drop_all) == peaks
    # and passthrough when classifier is None even if enabled
    cfg_on = Config(vision=VisionConfig(enabled=True))
    assert confirm_goals(peaks, {}, cfg_on, None) == peaks


def test_confirm_goals_prunes_with_stub():
    cfg = Config(vision=VisionConfig(enabled=True))
    peaks = [20.0, 40.0]
    frames_by_T = {
        20.0: ["/x/T20.0_0001.jpg", "/x/T20.0_0002.jpg"],
        40.0: ["/x/T40.0_0001.jpg", "/x/T40.0_0002.jpg"],
    }
    kept = confirm_goals(peaks, frames_by_T, cfg, _drop_early)
    assert kept == [40.0]                      # early goal pruned, order preserved


def test_confirm_goals_keeps_when_no_frames():
    cfg = Config(vision=VisionConfig(enabled=True))
    peaks = [15.0]
    # no frames extracted for this peak -> kept (can't judge, don't drop)
    assert confirm_goals(peaks, {15.0: []}, cfg, _drop_all) == [15.0]
