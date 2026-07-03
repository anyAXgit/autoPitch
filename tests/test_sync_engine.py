from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.sync_engine import compute_offsets


def test_recovers_known_offset(tmp_path):
    # camB's identical bursts occur 1.5s later in its own file than camA's.
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(10.0, 12.0, 10.0), (30.0, 32.0, 10.0)]},
        {"name": "camB", "color": "green", "offset": 1.5,
         "bursts": [(11.5, 13.5, 10.0), (31.5, 33.5, 10.0)]},
    ], duration=45.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    offsets = compute_offsets(res["audio"], ref_cam="camA")
    assert offsets["camA"] == 0.0
    assert abs(offsets["camB"] - 1.5) < 0.15   # within ~150ms


def test_single_cam_returns_zero(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(10, 12, 10)]},
    ], duration=20.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"), fps=30)
    assert compute_offsets(res["audio"], ref_cam="camA") == {"camA": 0.0}
