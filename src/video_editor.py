import os
import json
import subprocess


def _ffmpeg(args):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def _ensure_filters():
    """Verify the installed ffmpeg build supports the filters render_plan
    depends on (xfade/acrossfade for angle-cut crossfades). A stripped-down
    ffmpeg build silently lacks these, which otherwise surfaces as an opaque
    CalledProcessError deep into a multi-hour render. Fail fast instead."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        capture_output=True, text=True, check=True,
    )
    listing = out.stdout
    missing = [f for f in ("xfade", "acrossfade") if f not in listing]
    if missing:
        raise RuntimeError(
            "ffmpeg is missing the 'xfade'/'acrossfade' filters — "
            "install a full ffmpeg build"
        )


_HW_H264 = None


def _hw_available():
    """True when ffmpeg has the Apple VideoToolbox H.264 encoder (cached probe)."""
    global _HW_H264
    if _HW_H264 is None:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, check=True)
        _HW_H264 = "h264_videotoolbox" in out.stdout
    return _HW_H264


def h264_args(hw=True):
    """Encoder args for the H.264 outputs. VideoToolbox (Apple HW) measured ~2x
    faster than libx264 on this workload (HEVC decode dominates); falls back to
    libx264 when unavailable or when hw=False (config hw_encode)."""
    if hw and _hw_available():
        return ["-c:v", "h264_videotoolbox", "-b:v", "10M", "-allow_sw", "1",
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p"]


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def render_segment(seg, fps, out_path, width, height, vargs=None):
    # Input-side seek (`-ss` before `-i`) so cutting a short window out of a
    # long ORIGINAL source is a keyframe jump, not a full decode-from-zero
    # (modern ffmpeg still lands on the exact frame). Normalize to the render
    # canvas here (scale+pad) -- preprocess no longer transcodes video, so
    # this is where mixed cam resolutions become a common size for xfade.
    dur = seg["src_out"] - seg["src_in"]
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    _ffmpeg([
        "-ss", str(seg["src_in"]), "-i", seg["src"], "-t", str(dur),
        "-vf", vf, "-vsync", "cfr", "-r", str(fps),
        *(vargs or h264_args()),
        "-c:a", "aac", "-ar", "48000", out_path,
    ])


def render_clip(clip, fps, crossfade_sec, work_dir, out_path, width, height, vargs=None):
    os.makedirs(work_dir, exist_ok=True)
    vargs = vargs or h264_args()
    seg_paths = []
    for i, seg in enumerate(clip["segments"]):
        sp = os.path.join(work_dir, f"seg_{i}.mp4")
        render_segment(seg, fps, sp, width, height, vargs)
        seg_paths.append(sp)

    if len(seg_paths) == 1:
        _ffmpeg(["-i", seg_paths[0], "-c", "copy", out_path])
        return out_path

    first, second = seg_paths[0], seg_paths[1]
    d1 = probe_duration(first)
    off = max(0.0, d1 - crossfade_sec)
    _ffmpeg([
        "-i", first, "-i", second,
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration={crossfade_sec}:offset={off}[v];"
        f"[0:a][1:a]acrossfade=d={crossfade_sec}[a]",
        "-map", "[v]", "-map", "[a]",
        *vargs, "-c:a", "aac",
        out_path,
    ])
    return out_path


def mix_bgm(video_path, bgm_path, volume, out_path):
    """Mix a looped background-music track under a video's existing audio. The
    bgm is looped to cover the whole video and ducked to `volume`; the crowd
    audio stays at full level. Video stream is copied (no re-encode)."""
    _ffmpeg([
        "-i", video_path, "-stream_loop", "-1", "-i", bgm_path,
        "-filter_complex",
        f"[1:a]volume={volume}[bg];"
        f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-shortest", out_path,
    ])


def render_plan(plan, output_dir, bgm_path=None, bgm_volume=0.15, on_clip=None):
    """Render every clip in the plan. `on_clip(i, n, path)` (optional) is called
    after each clip finishes -- used by the GUI to report render progress."""
    _ensure_filters()
    os.makedirs(output_dir, exist_ok=True)
    fps = plan["fps"]
    xfade = plan["crossfade_sec"]
    width = plan.get("output_width", 1920)
    height = plan.get("output_height", 1080)
    vargs = h264_args(plan.get("hw_encode", True))
    clip_paths = []
    for i, clip in enumerate(plan["clips"]):
        name = f"highlight_{clip['T']:.1f}.mp4"
        out_path = os.path.join(output_dir, name)
        work = os.path.join(output_dir, f".work_{clip['T']:.1f}")
        render_clip(clip, fps, xfade, work, out_path, width, height, vargs)
        clip_paths.append(out_path)
        if on_clip:
            on_clip(i + 1, len(plan["clips"]), out_path)

    # write plan.json alongside outputs (editor contract / future UI)
    with open(os.path.join(output_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)

    # concat all clips into highlight_all.mp4
    if clip_paths:
        listfile = os.path.join(output_dir, "concat.txt")
        with open(listfile, "w") as f:
            for p in clip_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        all_path = os.path.join(output_dir, "highlight_all.mp4")
        _ffmpeg([
            "-f", "concat", "-safe", "0", "-i", listfile,
            *vargs, "-c:a", "aac",
            all_path,
        ])
        if bgm_path and os.path.exists(bgm_path):
            tmp = os.path.join(output_dir, "_with_bgm.mp4")
            mix_bgm(all_path, bgm_path, bgm_volume, tmp)
            os.replace(tmp, all_path)
    return clip_paths
