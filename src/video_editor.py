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
    """True when Apple VideoToolbox H.264 is listed and can actually encode.

    Some headless/test contexts expose `h264_videotoolbox` in `ffmpeg -encoders`
    but fail to create a compression session. Probe a tiny in-memory encode so
    render jobs can fall back to libx264 before doing real work.
    """
    global _HW_H264
    if _HW_H264 is None:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, check=True)
        if "h264_videotoolbox" not in out.stdout:
            _HW_H264 = False
        else:
            probe = subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=320x180:d=0.1:r=30",
                "-frames:v", "1",
                "-c:v", "h264_videotoolbox", "-b:v", "1M", "-allow_sw", "1",
                "-f", "null", "-",
            ], capture_output=True, text=True)
            _HW_H264 = probe.returncode == 0
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

    # goal_label burns a "GOAL" badge in a final overlay pass; when present the
    # assembled clip goes to a temp path first, then the overlay writes out_path.
    goal = bool(clip.get("goal_label"))
    assembled = os.path.join(work_dir, "assembled.mp4") if goal else out_path

    if len(seg_paths) == 1:
        # single segment: re-encode only if we still need the overlay pass, else copy
        _ffmpeg(["-i", seg_paths[0], "-c", "copy", assembled])
    elif crossfade_sec <= 0.01:
        # hard cut: segments share codec/canvas, so a re-encode concat is enough
        listfile = os.path.join(work_dir, "segs.txt")
        with open(listfile, "w") as f:
            for p in seg_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        _ffmpeg(["-f", "concat", "-safe", "0", "-i", listfile,
                 *vargs, "-c:a", "aac", assembled])
    else:
        current = seg_paths[0]
        for i, next_path in enumerate(seg_paths[1:], 1):
            last = i == len(seg_paths) - 1
            merged = assembled if last else os.path.join(work_dir, f"xfade_{i}.mp4")
            d1 = probe_duration(current)
            off = max(0.0, d1 - crossfade_sec)
            _ffmpeg([
                "-i", current, "-i", next_path,
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_sec}:offset={off}[v];"
                f"[0:a][1:a]acrossfade=d={crossfade_sec}[a]",
                "-map", "[v]", "-map", "[a]",
                *vargs, "-c:a", "aac",
                merged,
            ])
            current = merged

    if goal:
        png = os.path.join(work_dir, "goal.png")
        _goal_png(width, height, png)
        at = float(clip.get("goal_at", max(0.0, probe_duration(assembled) / 2)))
        _burn_goal(assembled, png, at, out_path, vargs)
    return out_path


def _goal_png(width, height, out_path):
    """Render a 'GOAL' badge to a transparent PNG (PIL) so overlay -- always
    available -- can composite it, sidestepping ffmpeg builds without drawtext.
    Sized relative to the canvas; cached per (w,h) by the caller."""
    from PIL import Image, ImageDraw, ImageFont
    text = "GOAL"
    fs = max(48, int(height * 0.14))
    font = None
    for fp in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
               "/Library/Fonts/Arial Unicode.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, fs); break
    if font is None:
        font = ImageFont.load_default()
    tmp = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(tmp)
    l, t, r, b = d.textbbox((0, 0), text, font=font, stroke_width=max(2, fs // 16))
    tw, th = r - l, b - t
    pad = int(fs * 0.4)
    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, img.width - 1, img.height - 1],
                        radius=int(fs * 0.25), fill=(10, 16, 24, 180))
    d.text((pad - l, pad - t), text, font=font, fill=(55, 214, 122, 255),
           stroke_width=max(2, fs // 16), stroke_fill=(4, 16, 31, 255))
    img.save(out_path)
    return out_path


def _burn_goal(video_path, png_path, at, out_path, vargs, hold=2.0, fade=0.3):
    """Overlay the GOAL badge (top-center) on the clip from `at` to `at+hold`
    with a quick fade in/out. Position is seconds-into-clip, so it's correct
    regardless of which segment the goal falls in."""
    end = at + hold
    _ffmpeg([
        "-i", video_path, "-i", png_path,
        "-filter_complex",
        f"[1:v]format=rgba,fade=t=in:st={at}:d={fade}:alpha=1,"
        f"fade=t=out:st={end - fade}:d={fade}:alpha=1[b];"
        f"[0:v][b]overlay=(W-w)/2:H*0.08:enable='between(t,{at},{end})'[v]",
        "-map", "[v]", "-map", "0:a?",
        *vargs, "-c:a", "aac", out_path,
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
