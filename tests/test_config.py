from src.config import load_config


def test_load_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("fps: 24\n")
    cfg = load_config(str(cfg_file))
    assert cfg.fps == 24              # from file
    assert cfg.build_up_sec == 5      # default
    assert cfg.max_clips is None      # nested default
    assert cfg.crossfade_sec == 0.5


def test_load_config_nested(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "peak:\n  threshold_k: 4.5\n  max_clips: 3\nreaction:\n  hold_sec: 1\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.threshold_k == 4.5
    assert cfg.max_clips == 3
    assert cfg.hold_sec == 1
    assert cfg.min_gap_sec == 6       # untouched nested default
