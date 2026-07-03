import os
from tests.make_dummy import make_dummy_set
from main import run


def test_e2e_multicam(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(20, 22, 10), (40, 42, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0,
         "bursts": [(20, 24, 16), (40, 44, 16)]},
    ], duration=60.0)
    monkeypatch.chdir(tmp_path)
    # write a config in tmp with small min_gap so both peaks survive
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 18,
        "crossfade_sec": 0.5, "peak": {"min_gap_sec": 10, "threshold_k": 2.0},
    }))
    result = run(raw_dir=str(raw), config_path=str(tmp_path / "config.yaml"))
    assert len(result["peaks"]) == 2
    assert len(result["clips"]) == 2
    for p in result["clips"]:
        assert os.path.exists(p)
    assert os.path.exists(os.path.join("data", "output", "highlight_all.mp4"))
    assert os.path.exists(os.path.join("data", "output", "plan.json"))


def test_e2e_singlecam(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(20, 22, 10), (40, 42, 10)]},
    ], duration=60.0)
    monkeypatch.chdir(tmp_path)
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 18,
        "crossfade_sec": 0.5, "peak": {"min_gap_sec": 10, "threshold_k": 2.0},
    }))
    result = run(raw_dir=str(raw), config_path=str(tmp_path / "config.yaml"))
    assert len(result["peaks"]) == 2
    assert len(result["clips"]) == 2
    for p in result["clips"]:
        assert os.path.exists(p)
    assert os.path.exists(os.path.join("data", "output", "highlight_all.mp4"))
    assert os.path.exists(os.path.join("data", "output", "plan.json"))

    # single-cam property: no angle switches, every clip has exactly one segment
    plan = result["plan"]
    for clip in plan["clips"]:
        assert len(clip["segments"]) == 1
        assert clip["segments"][0]["cam"] == "camA"


def test_e2e_main_cam_override(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    # "camA" sorts first alphabetically but is QUIET (weak burst only).
    # "camz" sorts after camA but has the REAL goal bursts.
    # Uppercase 'A' (65) < lowercase 'z' (122), so sorted() -> ["camA", "camz"],
    # making camA the default (wrong) choice unless main_cam override works.
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(25, 27, 3)]},
        {"name": "camz", "color": "green", "offset": 0.0,
         "bursts": [(20, 22, 12), (40, 42, 12)]},
    ], duration=55.0)
    monkeypatch.chdir(tmp_path)
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "main_cam": "camz",
        "build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 18,
        "crossfade_sec": 0.5, "peak": {"min_gap_sec": 10, "threshold_k": 2.0},
    }))
    result = run(raw_dir=str(raw), config_path=str(tmp_path / "config.yaml"))
    assert len(result["peaks"]) == 2
    for p in result["peaks"]:
        assert any(abs(p - t) <= 1.5 for t in (20, 40)), f"unexpected peak {p}"

    plan = result["plan"]
    for clip in plan["clips"]:
        assert clip["segments"][0]["cam"] == "camz"
