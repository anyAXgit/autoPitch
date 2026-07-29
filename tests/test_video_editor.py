import os
import subprocess
import pytest
from tests.make_dummy import make_dummy_set
from src.preprocess import preprocess_all
from src.config import load_config
from src.sync_engine import compute_offsets
from src.peak_detector import detect_peaks, rms_db
from src.segment_planner import build_plan
from src import video_editor
from src.video_editor import render_plan, probe_duration, mix_bgm


def _cfg(tmp_path):
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 18,
        "crossfade_sec": 0.5, "peak": {"min_gap_sec": 5, "threshold_k": 2.0},
    }))
    return load_config(str(p))


def _dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True).stdout.split()
    w, h = int(out[0]), int(out[1])
    return w, h


def test_render_plan_checks_ffmpeg_filters(tmp_path, monkeypatch):
    # A stripped-down ffmpeg build (missing xfade/acrossfade) must fail fast
    # and clearly at the top of render_plan, rather than mid-render with an
    # opaque CalledProcessError. Monkeypatch subprocess.run so the
    # `-filters` probe returns a filter listing without xfade/acrossfade,
    # and assert render_plan raises RuntimeError before doing any real
    # rendering work (no ffmpeg encode needed -- a minimal single-clip plan
    # with a nonexistent source path is enough, since the guard must raise
    # before render_segment ever touches it).
    class FakeCompletedProcess:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0

    def fake_run(args, **kwargs):
        # args[0] is a resolved path (a bundled binary in a packaged build),
        # so match on the program name rather than the literal string.
        assert (os.path.basename(args[0]).startswith("ffmpeg")
                and args[1:3] == ["-hide_banner", "-filters"]), (
            "test only stubs the -filters probe; real ffmpeg calls should "
            "never be reached because the guard must raise first"
        )
        # Realistic-looking filter listing, but with xfade/acrossfade
        # deliberately absent (as on a stripped ffmpeg build).
        listing = (
            " T.. concat           A+V->A+V   Concatenate audio and video streams.\n"
            " ... scale            V->V       Scale the input video size and/or convert the image format.\n"
            " ... fade             V->V       Fade in/out input video.\n"
        )
        return FakeCompletedProcess(listing)

    monkeypatch.setattr(video_editor.subprocess, "run", fake_run)

    minimal_plan = {
        "fps": 30,
        "crossfade_sec": 0.5,
        "clips": [
            {"T": 5.0, "segments": [
                {"cam": "camA", "src": str(tmp_path / "does_not_exist.mp4"),
                 "src_in": 0.0, "src_out": 5.0},
            ]},
        ],
    }
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="xfade"):
        render_plan(minimal_plan, str(out_dir))


def test_render_segment_applies_hflip_filter(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(video_editor, "_ffmpeg", lambda args: seen.setdefault("args", args))
    video_editor.render_segment({
        "src": "in.mp4",
        "src_in": 1.0,
        "src_out": 3.0,
        "hflip": True,
    }, 30, str(tmp_path / "out.mp4"), 1920, 1080, vargs=["-c:v", "libx264"])
    vf = seen["args"][seen["args"].index("-vf") + 1]
    assert vf.startswith("hflip,scale=")


def test_render_multicam_clip(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 18)]},
    ], duration=55.0)
    # small render canvas keeps the dummy render fast (normalization is now at render)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"),
                         fps=30, width=320, height=180)
    cfg = _cfg(tmp_path)
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    out_dir = tmp_path / "out"
    progress = []
    paths = render_plan(plan, str(out_dir),
                        on_clip=lambda i, n, p: progress.append((i, n)))
    assert len(paths) == 1
    assert progress == [(1, 1)]          # progress callback fired per clip
    assert os.path.exists(paths[0])
    assert os.path.exists(os.path.join(str(out_dir), "highlight_all.mp4"))
    # clip has audio + video streams
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", paths[0]], capture_output=True, text=True, check=True
    ).stdout
    assert "video" in streams and "audio" in streams
    # dynamic length is within clamp bounds
    # 2-segment clips render crossfade_sec shorter than the planned window (xfade overlaps content)
    assert 7.0 <= probe_duration(paths[0]) <= 19


def test_render_hard_cut_when_crossfade_zero(tmp_path):
    # crossfade_sec = 0 -> the angle switch is a HARD CUT (concat, no xfade):
    # output duration equals the full sum of segment lengths (no overlap eaten).
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 18)]},
    ], duration=55.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"),
                         fps=30, width=320, height=180)
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 18,
        "crossfade_sec": 0.0, "peak": {"min_gap_sec": 5, "threshold_k": 2.0},
    }))
    cfg = load_config(str(p))
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    clip = plan["clips"][0]
    assert len(clip["segments"]) == 2
    paths = render_plan(plan, str(tmp_path / "out"))
    span = sum(s["src_out"] - s["src_in"] for s in clip["segments"])
    assert abs(probe_duration(paths[0]) - span) < 0.5   # no crossfade overlap removed


def test_render_clip_with_three_angle_segments(tmp_path):
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "cam1", "color": "red", "bursts": [(5, 6, 10)]},
        {"name": "cam2", "color": "green", "bursts": [(5, 6, 10)]},
        {"name": "cam3", "color": "blue", "bursts": [(5, 6, 10)]},
    ], duration=12.0)
    plan = {
        "fps": 30,
        "crossfade_sec": 0.25,
        "output_width": 320,
        "output_height": 180,
        "hw_encode": False,
        "clips": [{
            "T": 5.0,
            "segments": [
                {"cam": "cam1", "src": str(raw / "cam1.mp4"), "src_in": 1.0, "src_out": 3.0},
                {"cam": "cam2", "src": str(raw / "cam2.mp4"), "src_in": 3.0, "src_out": 5.0},
                {"cam": "cam3", "src": str(raw / "cam3.mp4"), "src_in": 5.0, "src_out": 7.0},
            ],
        }],
    }
    paths = render_plan(plan, str(tmp_path / "out"))
    assert len(paths) == 1
    assert os.path.exists(paths[0])
    assert _dims(paths[0]) == (320, 180)
    assert 5.0 <= probe_duration(paths[0]) <= 6.1


def test_render_plan_mixes_bgm(tmp_path):
    # Rendering with a bgm file mixes music UNDER the highlight (video + audio
    # preserved, duration unchanged -- bgm is looped/ducked, not appended).
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0, "bursts": [(30, 34, 18)]},
    ], duration=55.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"),
                         fps=30, width=320, height=180)
    cfg = _cfg(tmp_path)
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)

    # short bgm; mix_bgm loops it to cover the clip
    bgm = str(tmp_path / "bgm.m4a")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:a", "aac", bgm], check=True)

    out_dir = tmp_path / "out"
    render_plan(plan, str(out_dir), bgm_path=bgm, bgm_volume=0.2)
    all_path = os.path.join(str(out_dir), "highlight_all.mp4")
    assert os.path.exists(all_path)
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", all_path], capture_output=True, text=True, check=True).stdout
    assert "video" in streams and "audio" in streams
    # duration preserved (music mixed under, clip length unchanged)
    dur = probe_duration(all_path)
    assert 7.0 <= dur <= 19


def test_mix_bgm_loops_short_track_to_video_end(tmp_path):
    import soundfile as sf

    video = str(tmp_path / "silent.mp4")
    bgm = str(tmp_path / "short_bgm.m4a")
    out = str(tmp_path / "mixed.mp4")
    tail = str(tmp_path / "tail.wav")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=160x90:d=5:r=30",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000:d=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", video,
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=1:sample_rate=48000",
        "-c:a", "aac", bgm,
    ], check=True)

    mix_bgm(video, bgm, 0.5, out)

    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", "4", "-t", "0.75", "-i", out,
        "-vn", "-ac", "1", "-ar", "16000", tail,
    ], check=True)
    y, _sr = sf.read(tail, dtype="float32")
    assert float((y ** 2).mean()) > 1e-5


def test_render_multicam_mixed_resolution(tmp_path):
    # Real-shoot hard case: cams start at genuinely different native
    # resolutions AND aspect ratios (e.g. DJI Pocket 4 vs. a smartphone).
    # H1's preprocess_all scale+pad normalization must land both cams on a
    # common canvas before build_plan/render_plan ever see them, otherwise
    # ffmpeg's xfade filter hard-fails on mismatched input dimensions.
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0,
         "size": "320x240", "bursts": [(30, 32, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0,
         "size": "640x360", "bursts": [(30, 34, 18)]},
    ], duration=55.0)
    res = preprocess_all(
        str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"),
        fps=30, width=320, height=180,
    )
    cfg = _cfg(tmp_path)
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)
    out_dir = tmp_path / "out"
    paths = render_plan(plan, str(out_dir))
    assert len(paths) == 1
    assert os.path.exists(paths[0])
    # clip has audio + video streams
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", paths[0]], capture_output=True, text=True, check=True
    ).stdout
    assert "video" in streams and "audio" in streams
    # mismatched-resolution inputs were normalized to the common canvas and
    # the xfade composited cleanly at that size (no dimension-mismatch error)
    assert _dims(paths[0]) == (320, 180)


def test_reaction_marker_lands_within_tolerance(tmp_path):
    # DoD-3 characterization test: at an angle cut (camA buildup -> camB
    # reaction xfade), a sharp audio marker placed at a known source time
    # in the reaction segment must land at the mathematically predicted
    # time in the rendered output, within a 0.1s A/V sync tolerance.
    #
    # Derivation: a 2-segment clip is buildup(camA,[start,cut]) xfaded with
    # reaction(camB,[seg1_in,seg1_out]) at offset = d0 - crossfade, where
    # d0 = cut - start is segment 0's actual length (the buildup now holds a
    # post_goal_sec beat past T before the cut, so d0 != build_up_sec) and
    # seg1_in = cut + offsets[camB] (NOTE:
    # make_dummy_set's per-cam "offset" field is *not* wired into the
    # pipeline anywhere -- real inter-camera offsets are always derived by
    # compute_offsets() via audio cross-correlation, so they must be read
    # back from the plan, not assumed to equal the literal dict value).
    # A marker authored at camB *source* time Tb sits (Tb - seg1_in) into
    # the reaction segment's own file, so in the xfade output it appears
    # at:
    #   expected = (d0 - crossfade) + (Tb - seg1_in)
    raw = tmp_path / "raw"
    Tb = 38.0
    make_dummy_set(str(raw), [
        # Long celebration on BOTH cams so (a) reaction_end extends the clip,
        # giving a wide reaction window even after the post-goal hold eats its
        # front, and (b) the two broad bursts share a shape so compute_offsets
        # locks to ~0 (a mismatched narrow/broad pair aliases to a spurious
        # offset). camA drives goal detection + reaction_end; camB carries the
        # sharp marker at Tb, placed well inside [cut, end].
        {"name": "camA", "color": "red", "offset": 0.0,
         "bursts": [(30, 44, 10)]},
        {"name": "camB", "color": "green", "offset": 0.0,
         # broad reaction burst so pick_reaction_angle picks camB, plus a
         # short, sharp, much louder marker burst at Tb that dominates the
         # reaction window and is trivially locatable via rms_db.
         "bursts": [(30, 44, 16), (Tb, Tb + 0.6, 30)]},
    ], duration=55.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"),
                         fps=30, width=320, height=180)
    # Large min_gap so the long single celebration collapses to ONE peak
    # (the 14s plateau would otherwise split into ~3 clusters at min_gap=5).
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "build_up_sec": 5, "min_len_sec": 8, "max_len_sec": 18,
        "crossfade_sec": 0.5, "peak": {"min_gap_sec": 30, "threshold_k": 2.0},
    }))
    cfg = load_config(str(cfg_path))
    offsets = compute_offsets(res["audio"], "camA")
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, offsets, peaks, "camA", cfg)

    assert len(plan["clips"]) == 1
    clip = plan["clips"][0]
    segments = clip["segments"]
    assert len(segments) == 2, (
        f"expected a 2-segment angle switch, got {len(segments)}: {segments}"
    )
    assert segments[0]["cam"] == "camA"
    assert segments[1]["cam"] == "camB"

    T_used = clip["T"]
    start = segments[0]["src_in"]
    seg1_in = segments[1]["src_in"]
    assert start == max(0.0, T_used - cfg.build_up_sec)

    out_dir = tmp_path / "out"
    paths = render_plan(plan, str(out_dir))
    assert len(paths) == 1

    # Extract mono 16kHz audio from the rendered clip and locate the
    # marker's onset time.
    wav_path = str(tmp_path / "marker.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", paths[0], "-vn", "-ac", "1", "-ar", "16000", wav_path],
        check=True,
    )
    # Use a small (0.05s) RMS window for tighter onset localization.
    times, db = rms_db(wav_path, 0.05)
    # The marker burst (gain 30) is much louder than everything else in the
    # mix, including the broad reaction burst (gain 16) it's embedded in --
    # its plateau sits near the track's max dB while the reaction burst's
    # own floor sits well below it. argmax() alone is ambiguous over a
    # flat-topped burst plateau (any frame on the plateau could "win" by
    # noise), so instead take the *onset*: the first frame within 3dB of
    # the peak. That isolates the marker's rising edge specifically,
    # rather than the broader reaction burst's onset or an arbitrary point
    # on the marker's plateau.
    peak_db = float(db.max())
    onset_idx = next(i for i, d in enumerate(db) if d >= peak_db - 3.0)
    measured_marker_time = float(times[onset_idx])

    # NOTE: `d0` here is the *planned* segment-0 length (T_used - start); the
    # renderer's xfade offset actually uses `d1 = probe_duration(seg_0)`, the
    # encoded file's real duration. CFR framing (1/30s) and AAC encoder delay
    # make d1 differ from d0 by a few tens of ms, so the tolerance below is a
    # measurement budget (0.2s), deliberately looser than the product's <0.1s
    # sync target: a genuine render-offset bug shifts the marker by a full
    # crossfade (~0.5s), which still fails this comfortably.
    d0 = segments[0]["src_out"] - start   # actual buildup length (start..cut)
    off = d0 - cfg.crossfade_sec
    expected = off + (Tb - seg1_in)
    diff = abs(measured_marker_time - expected)
    assert diff <= 0.2, (
        f"A/V sync miss at angle cut: measured={measured_marker_time:.3f}s "
        f"expected={expected:.3f}s diff={diff:.3f}s "
        f"(T_used={T_used}, start={start}, seg1_in={seg1_in}, "
        f"crossfade={cfg.crossfade_sec})"
    )


def test_render_goal_label_overlays_and_preserves_duration(tmp_path):
    # A clip flagged goal_label renders (GOAL badge burned via PNG overlay) and
    # keeps its source duration -- the overlay pass must not drop/extend frames.
    raw = tmp_path / "raw"
    make_dummy_set(str(raw), [
        {"name": "camA", "color": "red", "offset": 0.0, "bursts": [(30, 32, 12)]},
    ], duration=50.0)
    res = preprocess_all(str(raw), str(tmp_path / "tv"), str(tmp_path / "ta"),
                         fps=30, width=320, height=180)
    cfg = _cfg(tmp_path)
    peaks = detect_peaks(res["audio"]["camA"], cfg)
    plan = build_plan(res, {"camA": 0.0}, peaks, "camA", cfg)
    clip = plan["clips"][0]
    span = sum(s["src_out"] - s["src_in"] for s in clip["segments"])

    plain = render_plan(plan, str(tmp_path / "plain"))
    d_plain = probe_duration(plain[0])

    clip["goal_label"] = True
    clip["goal_at"] = 3.0
    labeled = render_plan(plan, str(tmp_path / "labeled"))
    assert os.path.exists(labeled[0])
    assert abs(probe_duration(labeled[0]) - d_plain) < 0.3   # overlay kept length
    assert abs(probe_duration(labeled[0]) - span) < 0.6
