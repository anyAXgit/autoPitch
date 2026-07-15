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


def test_confirm_goals_confidence_threshold():
    # is_goal True but low confidence -> pruned when min_confidence is high
    peaks = [10.0, 20.0]
    frames_by_T = {10.0: ["/x/T10.0_0001.jpg"], 20.0: ["/x/T20.0_0001.jpg"]}

    def low_then_high(frames):
        import os
        t = float(os.path.basename(frames[0])[1:].split("_")[0])
        return {"is_goal": True, "confidence": 0.6 if t < 15 else 0.95}

    cfg = Config(vision=VisionConfig(enabled=True, min_confidence=0.9))
    assert confirm_goals(peaks, frames_by_T, cfg, low_then_high) == [20.0]
    # threshold off (0.0) keeps both
    cfg0 = Config(vision=VisionConfig(enabled=True, min_confidence=0.0))
    assert confirm_goals(peaks, frames_by_T, cfg0, low_then_high) == [10.0, 20.0]


def test_confirm_goals_keeps_when_no_frames():
    cfg = Config(vision=VisionConfig(enabled=True))
    peaks = [15.0]
    # no frames extracted for this peak -> kept (can't judge, don't drop)
    assert confirm_goals(peaks, {15.0: []}, cfg, _drop_all) == [15.0]


def test_confirm_roi_clips_prunes_only_roi_only(monkeypatch, tmp_path):
    # Audio-backed clips pass untouched; ROI-only clips are judged by the
    # net-crop classifier and dropped when it says not-a-goal.
    from src.config import Config, LocateConfig
    from src import goal_confirmer as gc

    monkeypatch.setattr(gc, "net_crops", lambda *a, **k: ["/x/net_1.jpg"])
    import src.goal_locator as gl
    monkeypatch.setattr(gl, "roi_for_cam", lambda *a, **k: [0, 0, 1, 1])

    plan = {"clips": [
        {"T": 10.0, "roi_only": False, "goal_cam": None,
         "segments": [{"cam": "camA", "src": "/x/a.mp4"}]},
        {"T": 50.0, "roi_only": True, "goal_cam": "camB",
         "segments": [{"cam": "camB", "src": "/x/b.mp4"}]},
        {"T": 90.0, "roi_only": True, "goal_cam": "camB",
         "segments": [{"cam": "camB", "src": "/x/b.mp4"}]},
    ]}
    verdicts = {50.0: {"is_goal": True, "confidence": 0.9},
                90.0: {"is_goal": False, "confidence": 0.8}}
    state = {"i": [50.0, 90.0]}

    def classifier(crops):
        return verdicts[state["i"].pop(0)]

    cfg = Config(locate=LocateConfig(enabled=True))
    out = gc.confirm_roi_clips(plan, cfg, {"camB": [0, 0, 1, 1]}, {"camB": 0.0},
                               classifier, str(tmp_path))
    ts = [c["T"] for c in out["clips"]]
    assert ts == [10.0, 50.0]                      # 90.0 pruned by the judge
    assert out["clips"][1]["roi_verdict"]["is_goal"] is True
