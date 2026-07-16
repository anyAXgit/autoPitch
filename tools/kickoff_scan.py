"""Prototype: detect kickoff/restart states in the WIDE view (quiet-goal cue).

Every goal forces a restart: players drift back to their own halves and play
resumes from the center. Unlike the net region (measured dead end: oblique
angle, occlusions, resident spare balls), this signal is people-sized, lasts
seconds, and is structural rather than appearance-based.

v0 state definition, per sampled frame (1 fps):
  - detect persons (YOLO nano, imgsz=1280 for small far-side players)
  - low global motion (median |frame diff| under threshold — walking, not play)
  - occupancy asymmetry: min(left, right) / total below a threshold, i.e. one
    half nearly empty while players regroup (the measured post-goal signature)
A restart EVENT = the state holding for >= min_hold seconds. Candidate goal
time = event start - lead_sec.

Validation mode: compare detected events against known audio goal times.
Run: ./.venv/bin/python tools/kickoff_scan.py <video> [--goals t1,t2,...]
"""
import sys, subprocess, json
import numpy as np

W, H_FPS = 640, 1.0     # decode width / sample rate
MOTION_MAX = 4.5        # mean abs diff (0-255) below this = play paused
ASYM_MAX = 0.18         # min(left,right)/total below this = one half empty
MIN_PEOPLE = 4          # need enough players visible to judge occupancy
MIN_HOLD = 3            # seconds the state must persist
MERGE_GAP = 12          # events closer than this merge (one restart)
LEAD_SEC = 12.0         # goal happens ~this long before the restart state


def decode_wide(video, fps=H_FPS, width=W):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-i", video,
         "-vf", f"fps={fps},scale={width}:-2", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", video],
        capture_output=True, text=True).stdout
    st = json.loads(probe)["streams"][0]
    h = int(round(width * st["height"] / st["width"])); h -= h % 2
    n = len(out) // (width * h * 3)
    return np.frombuffer(out, np.uint8)[:n * width * h * 3].reshape(n, h, width, 3)


def person_xs(model, frames, batch=16):
    """Per-frame person center-x lists."""
    res = []
    for i in range(0, len(frames), batch):
        rs = model.predict(list(frames[i:i + batch]), classes=[0], conf=0.25,
                           imgsz=1280, verbose=False)
        for r in rs:
            res.append([float((b.xyxy[0][0] + b.xyxy[0][2]) / 2) for b in r.boxes])
    return res


def restart_events(frames, xs_per_frame):
    gray = frames.mean(axis=3)
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))   # per second
    motion = np.concatenate([[motion[0]], motion])
    state = []
    for i, xs in enumerate(xs_per_frame):
        if len(xs) < MIN_PEOPLE or motion[i] > MOTION_MAX:
            state.append(False); continue
        left = sum(1 for x in xs if x < W * 0.5)
        right = len(xs) - left
        state.append(min(left, right) / len(xs) <= ASYM_MAX)
    # runs of True >= MIN_HOLD
    events, start = [], None
    for i, s in enumerate(state + [False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= MIN_HOLD:
                events.append(float(start))
            start = None
    merged = []
    for t in events:
        if merged and t - merged[-1] < MERGE_GAP:
            continue
        merged.append(t)
    return merged, state, motion


def main():
    video = sys.argv[1]
    goals = []
    if "--goals" in sys.argv:
        goals = [float(x) for x in sys.argv[sys.argv.index("--goals") + 1].split(",")]
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("decoding...", flush=True)
    frames = decode_wide(video)
    print(f"{len(frames)} frames @1fps; detecting persons...", flush=True)
    xs = person_xs(model, frames)
    events, state, motion = restart_events(frames, xs)
    print(f"\n재개 이벤트 {len(events)}개:")
    for t in events:
        cand = max(0.0, t - LEAD_SEC)
        near = min(goals, key=lambda g: abs(g - cand)) if goals else None
        tag = ""
        if goals:
            d = cand - near
            tag = f"  → 후보골 {int(cand//60)}:{cand%60:04.1f}  (가까운 실제골 {int(near//60)}:{near%60:04.1f}, Δ{d:+.0f}s)"
        print(f"  재개 {int(t//60)}:{t%60:04.1f}{tag}")
    if goals:
        matched = sum(1 for g in goals
                      if any(abs((t - LEAD_SEC) - g) <= 20 for t in events))
        print(f"\n실제 골 {len(goals)}개 중 재개로 커버된 것: {matched}개 "
              f"(±20s) | 잉여 이벤트: {max(0, len(events) - matched)}개")


if __name__ == "__main__":
    main()
