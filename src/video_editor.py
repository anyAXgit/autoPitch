import os
import json
import subprocess


def _ffmpeg(args):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def render_segment(seg, fps, out_path):
    # accurate seek: -ss/-to after -i, re-encode to CFR
    _ffmpeg([
        "-i", seg["src"], "-ss", str(seg["src_in"]), "-to", str(seg["src_out"]),
        "-vsync", "cfr", "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", out_path,
    ])


def render_clip(clip, fps, crossfade_sec, work_dir, out_path):
    os.makedirs(work_dir, exist_ok=True)
    seg_paths = []
    for i, seg in enumerate(clip["segments"]):
        sp = os.path.join(work_dir, f"seg_{i}.mp4")
        render_segment(seg, fps, sp)
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
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        out_path,
    ])
    return out_path


def render_plan(plan, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fps = plan["fps"]
    xfade = plan["crossfade_sec"]
    clip_paths = []
    for clip in plan["clips"]:
        name = f"highlight_{clip['T']:.1f}.mp4"
        out_path = os.path.join(output_dir, name)
        work = os.path.join(output_dir, f".work_{clip['T']:.1f}")
        render_clip(clip, fps, xfade, work, out_path)
        clip_paths.append(out_path)

    # write plan.json alongside outputs (editor contract / future UI)
    with open(os.path.join(output_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)

    # concat all clips into highlight_all.mp4
    if clip_paths:
        listfile = os.path.join(output_dir, "concat.txt")
        with open(listfile, "w") as f:
            for p in clip_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        _ffmpeg([
            "-f", "concat", "-safe", "0", "-i", listfile,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            os.path.join(output_dir, "highlight_all.mp4"),
        ])
    return clip_paths
