#!/usr/bin/env python3
"""Regenerate the app icons from packaging/icon/icon.png.

    python packaging/make_icons.py

Writes autoPitch.icns (macOS) and autoPitch.ico (Windows) next to the source.
Both are committed, so this only needs running when the artwork changes -- a
Windows build has no way to produce an .icns, and vice versa.

macOS needs `iconutil`, which ships with the Xcode command line tools. The .ico
is written by Pillow and works anywhere.
"""
import os
import shutil
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(HERE, "icon")
SRC = os.path.join(ICON_DIR, "icon.png")

# macOS asks for each size at 1x and 2x; Windows takes them all in one file.
MAC_POINTS = (16, 32, 128, 256, 512)
WIN_SIZES = (16, 24, 32, 48, 64, 128, 256)


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"원본이 없습니다: {SRC}")
    im = Image.open(SRC).convert("RGBA")
    if im.width != im.height:
        raise SystemExit(f"정사각형이어야 합니다: {im.width}x{im.height}")
    # Everything is resampled from one 1024px master rather than from each
    # other, so a small size never inherits a previous resize's softness.
    base = im.resize((1024, 1024), Image.LANCZOS)

    ico = os.path.join(ICON_DIR, "autoPitch.ico")
    base.save(ico, format="ICO", sizes=[(s, s) for s in WIN_SIZES])
    print(f"  {os.path.basename(ico)}  ({os.path.getsize(ico) / 1024:.0f} KB)")

    if sys.platform != "darwin":
        print("  autoPitch.icns 는 macOS 에서만 만들 수 있습니다 — 건너뜁니다.")
        return
    if not shutil.which("iconutil"):
        raise SystemExit("iconutil 이 없습니다. Xcode 명령줄 도구를 설치해 주세요:\n"
                         "    xcode-select --install")
    iconset = os.path.join(ICON_DIR, "autoPitch.iconset")
    shutil.rmtree(iconset, ignore_errors=True)
    os.makedirs(iconset)
    try:
        for pt in MAC_POINTS:
            for scale in (1, 2):
                suffix = "@2x" if scale == 2 else ""
                px = pt * scale
                base.resize((px, px), Image.LANCZOS).save(
                    os.path.join(iconset, f"icon_{pt}x{pt}{suffix}.png"))
        icns = os.path.join(ICON_DIR, "autoPitch.icns")
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
        print(f"  {os.path.basename(icns)}  ({os.path.getsize(icns) / 1024:.0f} KB)")
    finally:
        shutil.rmtree(iconset, ignore_errors=True)


if __name__ == "__main__":
    main()
