import os

from src.config import load_config
from src.preprocess import preprocess_all
from src.sync_engine import compute_offsets
from src.peak_detector import detect_peaks
from src.segment_planner import build_plan
from src.video_editor import render_plan
from src.frame_extractor import extract_goal_frames
from src.goal_confirmer import confirm_goals, make_vlm_classifier

TEMP_VIDEO = "data/temp_video"
TEMP_AUDIO = "data/temp_audio"
TEMP_FRAMES = "data/temp_frames"
OUTPUT_DIR = "data/output"


def run(
    raw_dir="data/raw",
    config_path="config.yaml",
    temp_video=TEMP_VIDEO,
    temp_audio=TEMP_AUDIO,
    output_dir=OUTPUT_DIR,
    temp_frames=TEMP_FRAMES,
    vision_classifier=None,
):
    cfg = load_config(config_path)
    print(f"[1/5] preprocess: {raw_dir}")
    pre = preprocess_all(
        raw_dir, temp_video, temp_audio, cfg.fps,
        cfg.output_width, cfg.output_height,
    )
    if cfg.main_cam and cfg.main_cam not in pre["cams"]:
        print(f"[warn] main_cam '{cfg.main_cam}' not found in {pre['cams']}; using {pre['cams'][0]}")
    camA = cfg.main_cam if (cfg.main_cam and cfg.main_cam in pre["cams"]) else pre["cams"][0]
    print(f"      cams={pre['cams']} multicam={pre['is_multicam']}")

    if pre["is_multicam"]:
        print("[2/5] sync engine")
        offsets = compute_offsets(pre["audio"], camA)
        for c, o in offsets.items():
            print(f"      {c}: {o:+.3f}s")
    else:
        offsets = {camA: 0.0}
        print("[2/5] single-cam: skip sync")

    print("[3/5] peak detection")
    peaks = detect_peaks(pre["audio"][camA], cfg)
    print(f"      {len(peaks)} candidate(s): {[round(p,1) for p in peaks]}")

    if cfg.vision.enabled and peaks:
        frames_by_T = {}
        for T in peaks:
            frames_by_T[T] = extract_goal_frames(
                pre["source"][camA], T, cfg,
                os.path.join(temp_frames, f"T{T:.1f}"),
            )
        classifier = vision_classifier or make_vlm_classifier(cfg)
        kept = confirm_goals(peaks, frames_by_T, cfg, classifier)
        print(f"      vision: kept {len(kept)}/{len(peaks)} goals")
        peaks = kept

    print("[4/5] planning")
    plan = build_plan(pre, offsets, peaks, camA, cfg)

    print("[5/5] rendering")
    clips = render_plan(plan, output_dir)
    for c in clips:
        print(f"      -> {c}")
    print(f"done: {len(clips)} clip(s) + highlight_all.mp4 in {output_dir}")
    return {"peaks": peaks, "clips": clips, "plan": plan}


if __name__ == "__main__":
    run()
