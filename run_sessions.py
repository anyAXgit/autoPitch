#!/usr/bin/env python3
"""Pair real 2-cam footage (data/raw/cam1 + data/raw/cam2) into games by
capture time and run the highlight pipeline once per game.

The base pipeline treats a raw dir as ONE session of N cams. Here the shoot is
laid out as cam1/ (DJI Pocket, center = Cam A) and cam2/ (phone, corner = sub),
each holding several games. This driver matches each cam1 clip to its cam2
partner by `creation_time`, stages the pair into a per-game dir (DJI sorts
first alphabetically -> becomes Cam A automatically), and calls main.run()
with per-game temp/output dirs so nothing collides.

Usage:
    ./.venv/bin/python run_sessions.py --list        # show pairing, no render
    ./.venv/bin/python run_sessions.py --game 1      # render game 1 only
    ./.venv/bin/python run_sessions.py --all         # render every game
"""
import argparse
import os
import subprocess
from datetime import datetime, timezone

from main import run
from src.console import enable_utf8

enable_utf8()

CAM1_DIR = "data/raw/cam1"
CAM2_DIR = "data/raw/cam2"
STAGE_DIR = "data/_stage"
VIDEO_EXTS = (".mp4", ".mov", ".m4v")


def _list_videos(d):
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )


def _probe(path):
    """Return (start_epoch_seconds_or_None, duration_seconds)."""
    def q(entries):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", entries,
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().splitlines()

    dur = float(q("format=duration")[0])
    ct = q("format_tags=creation_time")
    start = None
    if ct and ct[0]:
        start = datetime.fromisoformat(ct[0].replace("Z", "+00:00")).timestamp()
    return start, dur


def pair_games(cam1_dir=CAM1_DIR, cam2_dir=CAM2_DIR):
    """Match each cam1 clip to its nearest-in-time cam2 partner.

    Returns a list of dicts (game 1..N in cam1 chronological order):
    {game, cam1, cam2, start1, start2, dur1, dur2, overlap}.
    """
    c1 = [{"path": p, "start": s, "dur": d}
          for p in _list_videos(cam1_dir) for s, d in [_probe(p)]]
    c2 = [{"path": p, "start": s, "dur": d}
          for p in _list_videos(cam2_dir) for s, d in [_probe(p)]]
    if any(r["start"] is None for r in c1 + c2):
        raise SystemExit("A clip is missing creation_time metadata; cannot pair by time.")

    c1.sort(key=lambda r: r["start"])
    remaining = list(c2)
    games = []
    for i, a in enumerate(c1, 1):
        b = min(remaining, key=lambda r: abs(r["start"] - a["start"]))
        remaining.remove(b)
        s = max(a["start"], b["start"])
        e = min(a["start"] + a["dur"], b["start"] + b["dur"])
        games.append({
            "game": i, "cam1": a["path"], "cam2": b["path"],
            "start1": a["start"], "start2": b["start"],
            "dur1": a["dur"], "dur2": b["dur"], "overlap": max(0.0, e - s),
        })
    return games


def _fmt(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).astimezone().strftime("%H:%M:%S")


def print_pairing(games):
    print(f"{'game':>4}  {'cam1 (Cam A)':<34} {'start':>8} {'len':>6}   "
          f"{'cam2 (sub)':<14} {'start':>8} {'len':>6}   {'overlap':>8}")
    for g in games:
        print(f"{g['game']:>4}  {os.path.basename(g['cam1']):<34} "
              f"{_fmt(g['start1']):>8} {g['dur1']/60:>5.1f}m   "
              f"{os.path.basename(g['cam2']):<14} "
              f"{_fmt(g['start2']):>8} {g['dur2']/60:>5.1f}m   "
              f"{g['overlap']/60:>6.1f}m")
        if g["overlap"] < 60:
            print(f"       !! low overlap ({g['overlap']:.0f}s) — sync may be unreliable")


def _stage(game):
    d = os.path.join(STAGE_DIR, f"game{game['game']}")
    os.makedirs(d, exist_ok=True)
    for src in (game["cam1"], game["cam2"]):
        link = os.path.join(d, os.path.basename(src))
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src), link)
    return d


def run_game(game):
    g = game["game"]
    print(f"\n===== GAME {g}: {os.path.basename(game['cam1'])} + "
          f"{os.path.basename(game['cam2'])} (overlap {game['overlap']/60:.1f}m) =====")
    stage_dir = _stage(game)
    return run(
        raw_dir=stage_dir,
        temp_video=f"data/temp_video/game{g}",
        temp_audio=f"data/temp_audio/game{g}",
        output_dir=f"data/output/game{g}",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--list", action="store_true", help="show pairing only")
    grp.add_argument("--game", type=int, metavar="N", help="render game N (1-based)")
    grp.add_argument("--all", action="store_true", help="render every game")
    args = ap.parse_args()

    games = pair_games()
    print_pairing(games)
    if args.list:
        return
    if args.game:
        match = [g for g in games if g["game"] == args.game]
        if not match:
            raise SystemExit(f"No game {args.game} (have 1..{len(games)}).")
        run_game(match[0])
    else:
        for g in games:
            run_game(g)


if __name__ == "__main__":
    main()
