import os
import subprocess

# Recognized raw video containers (matched case-insensitively).
_VIDEO_EXTS = (".mp4", ".mov", ".m4v")


def _ffmpeg(args):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def list_raw(raw_dir):
    return sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if os.path.splitext(f)[1].lower() in _VIDEO_EXTS
    )


def cam_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def preprocess_all(raw_dir, temp_video_dir, temp_audio_dir, fps, width=1920, height=1080):
    """Extract a 16kHz mono analysis WAV per cam. Video is NOT re-encoded here
    -- the original source is kept and normalized per-segment at render time
    (see video_editor.render_segment), so long footage isn't fully transcoded
    up front. `width`/`height`/`fps` are the render canvas, passed through for
    build_plan to stamp into plan.json. `temp_video_dir` is unused (kept for
    call-site compatibility)."""
    os.makedirs(temp_audio_dir, exist_ok=True)
    raws = list_raw(raw_dir)
    if not raws:
        raise FileNotFoundError(f"No video files in {raw_dir}")
    cams, source, audio = [], {}, {}
    for path in raws:
        cid = cam_id(path)
        cams.append(cid)
        aout = os.path.join(temp_audio_dir, f"{cid}.wav")
        _ffmpeg([
            "-i", path, "-vn", "-ac", "1", "-ar", "16000", aout,
        ])
        source[cid] = path
        audio[cid] = aout
    return {
        "cams": cams,
        "source": source,
        "audio": audio,
        "is_multicam": len(cams) > 1,
        "width": width,
        "height": height,
        "fps": fps,
    }
