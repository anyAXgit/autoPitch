import os
import glob
import subprocess
from src.ffmpeg import ffmpeg


def _ffmpeg(args):
    subprocess.run(
        [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def extract_goal_frames(source_path, T, cfg, out_dir, offset=0.0):
    """Extract a sparse set of downscaled JPG frames around a candidate goal
    for vision confirmation (V3). Window is [T-pre_sec, T+post_sec] on the Cam A
    timeline, shifted by `offset` into a sub-cam's own source timeline. Sampling
    at `vision.fps` keeps this cheap -- decode-only (no video encode), and only
    a few frames per goal, not the whole match.

    Returns the sorted list of written JPG paths.
    """
    v = cfg.vision
    start = max(0.0, T + offset - v.pre_sec)
    dur = v.pre_sec + v.post_sec
    os.makedirs(out_dir, exist_ok=True)
    # `-ss` before `-i` = fast keyframe seek into a long source. `fps` samples
    # the window; `scale=-2:H` fixes height, auto (even) width, aspect kept.
    vf = f"fps={v.fps},scale=-2:{v.frame_height}"
    pattern = os.path.join(out_dir, f"T{T:.1f}_%04d.jpg")
    _ffmpeg([
        "-ss", str(start), "-i", source_path, "-t", str(dur),
        "-vf", vf, pattern,
    ])
    return sorted(glob.glob(os.path.join(out_dir, f"T{T:.1f}_*.jpg")))
