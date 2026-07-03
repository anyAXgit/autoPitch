import os
import subprocess


def _ffmpeg(args):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def _volume_expr(bursts):
    # base quiet sine, boosted inside each burst window
    enables = "+".join(f"between(t,{s},{e})" for s, e, _g in bursts) or "0"
    # use the max gain across bursts is not needed; apply per-burst via chained volume
    chain = ["volume=0.1"]
    for s, e, g in bursts:
        chain.append(f"volume=enable='between(t,{s},{e})':volume={g}")
    return ",".join(chain), enables


def make_dummy_set(out_dir, cams, duration=45.0, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for cam in cams:
        name = cam["name"]
        path = os.path.join(out_dir, f"{name}.mp4")
        afilter, _ = _volume_expr(cam.get("bursts", []))
        _ffmpeg([
            "-f", "lavfi", "-i",
            f"color=c={cam['color']}:s=320x240:d={duration}:r={fps}",
            "-f", "lavfi", "-i",
            f"sine=frequency=200:duration={duration}:sample_rate=44100",
            "-af", afilter,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            path,
        ])
        files[name] = path
    return {"files": files, "cams": cams, "duration": duration, "fps": fps}
