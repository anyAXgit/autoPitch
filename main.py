from src.config import load_config
from src.preprocess import preprocess_all
from src.sync_engine import compute_offsets
from src.peak_detector import detect_peaks
from src.segment_planner import build_plan
from src.video_editor import render_plan

TEMP_VIDEO = "data/temp_video"
TEMP_AUDIO = "data/temp_audio"
OUTPUT_DIR = "data/output"


def run(raw_dir="data/raw", config_path="config.yaml"):
    cfg = load_config(config_path)
    print(f"[1/5] preprocess: {raw_dir}")
    pre = preprocess_all(raw_dir, TEMP_VIDEO, TEMP_AUDIO, cfg.fps)
    camA = pre["cams"][0]
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
    print(f"      {len(peaks)} goal(s): {[round(p,1) for p in peaks]}")

    print("[4/5] planning")
    plan = build_plan(pre, offsets, peaks, camA, cfg)

    print("[5/5] rendering")
    clips = render_plan(plan, OUTPUT_DIR)
    for c in clips:
        print(f"      -> {c}")
    print(f"done: {len(clips)} clip(s) + highlight_all.mp4 in {OUTPUT_DIR}")
    return {"peaks": peaks, "clips": clips, "plan": plan}


if __name__ == "__main__":
    run()
