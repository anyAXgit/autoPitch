"""config.yaml must survive being edited from the UI.

The point of src/config_edit is that comments carry measured reasoning that a
yaml round-trip would silently delete, so these tests care as much about what
is left untouched as about what changes.
"""
import pytest

from src.config_edit import read_values, set_values

SAMPLE = """\
fps: 30
build_up_sec: 5
peak:
  rms_window_sec: 0.5
  threshold_k: 3.0     # lower = more candidates
  max_clips: null
hw_encode: true            # Apple VideoToolbox
locate:
  enabled: true
  scan_enabled: false      # measured 2026-07-16: no separation at this venue
  nested:
    deep_value: 7
"""


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


def test_reads_nested_and_typed_values(cfg):
    got = read_values(cfg, ["fps", "peak.threshold_k", "peak.max_clips",
                            "hw_encode", "locate.scan_enabled",
                            "locate.nested.deep_value"])
    assert got == {"fps": 30, "peak.threshold_k": 3.0, "peak.max_clips": None,
                   "hw_encode": True, "locate.scan_enabled": False,
                   "locate.nested.deep_value": 7}


def test_same_key_name_at_two_levels_is_not_confused(cfg):
    # `enabled` exists only under locate here, but the dotted path must be what
    # resolves it -- a bare-name match would be ambiguous in the real file.
    assert read_values(cfg, ["locate.enabled"]) == {"locate.enabled": True}
    assert read_values(cfg, ["enabled"]) == {}


def test_edit_preserves_comments_and_untouched_lines(cfg):
    changed, missing = set_values(cfg, {"peak.threshold_k": 2.5,
                                        "locate.scan_enabled": True})
    assert sorted(changed) == ["locate.scan_enabled", "peak.threshold_k"]
    assert missing == []
    text = cfg.read_text(encoding="utf-8")
    assert "# lower = more candidates" in text
    assert "# measured 2026-07-16: no separation at this venue" in text
    assert "# Apple VideoToolbox" in text
    # the comment must stay in its original column, not reflow
    col = lambda t, k: next(l.index("#") for l in t.splitlines()
                            if l.strip().startswith(k + ":") and "#" in l)
    assert col(text, "threshold_k") == col(SAMPLE, "threshold_k")
    assert col(text, "scan_enabled") == col(SAMPLE, "scan_enabled")
    assert "threshold_k: 2.5" in text and "scan_enabled: true" in text
    # everything else identical
    before, after = SAMPLE.splitlines(), text.splitlines()
    assert len(before) == len(after)
    diff = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(diff) == 2


def test_roundtrip_types(cfg):
    set_values(cfg, {"hw_encode": False, "peak.max_clips": 12, "fps": 60})
    assert read_values(cfg, ["hw_encode", "peak.max_clips", "fps"]) == {
        "hw_encode": False, "peak.max_clips": 12, "fps": 60}


def test_unknown_key_is_reported_not_appended(cfg):
    changed, missing = set_values(cfg, {"peak.nope": 1})
    assert changed == [] and missing == ["peak.nope"]
    assert cfg.read_text(encoding="utf-8") == SAMPLE


def test_unchanged_value_does_not_rewrite(cfg):
    mtime = cfg.stat().st_mtime_ns
    changed, _ = set_values(cfg, {"fps": 30})
    assert changed == []
    assert cfg.stat().st_mtime_ns == mtime
