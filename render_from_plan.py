#!/usr/bin/env python3
"""Re-render a highlight from an (edited) plan.json — closes the editing loop.

The editor UI (editor/index.html) exports an edited plan.json (clips dropped,
reordered, or cut points nudged). Feed it here to render the new highlight
without re-running detection.

Usage:
    ./.venv/bin/python render_from_plan.py <plan.json> <output_dir> \
        [--bgm data/bgm.mp3] [--bgm-volume 0.15]
"""
import argparse
import json

from src.video_editor import render_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", help="path to an (edited) plan.json")
    ap.add_argument("output_dir", help="where to write highlight_*.mp4 + highlight_all.mp4")
    ap.add_argument("--bgm", default=None, help="optional background-music file")
    ap.add_argument("--bgm-volume", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    if not plan.get("clips"):
        raise SystemExit("plan has no clips to render.")
    clips = render_plan(plan, args.output_dir, args.bgm, args.bgm_volume)
    print(f"rendered {len(clips)} clip(s) + highlight_all.mp4 in {args.output_dir}")


if __name__ == "__main__":
    main()
