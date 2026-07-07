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
    os.makedirs(temp_video_dir, exist_ok=True)
    os.makedirs(temp_audio_dir, exist_ok=True)
    raws = list_raw(raw_dir)
    if not raws:
        raise FileNotFoundError(f"No .mp4 files in {raw_dir}")
    cams, video, audio = [], {}, {}
    for path in raws:
        cid = cam_id(path)
        cams.append(cid)
        vout = os.path.join(temp_video_dir, f"{cid}.mp4")
        aout = os.path.join(temp_audio_dir, f"{cid}.wav")
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
        _ffmpeg([
            "-i", path, "-vf", vf, "-vsync", "cfr", "-r", str(fps),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", vout,
        ])
        _ffmpeg([
            "-i", path, "-vn", "-ac", "1", "-ar", "16000", aout,
        ])
        video[cid] = vout
        audio[cid] = aout
    return {
        "cams": cams,
        "video": video,
        "audio": audio,
        "is_multicam": len(cams) > 1,
    }
