#!/usr/bin/env python3
"""Build the macOS .app / Windows .exe.

    python packaging/build.py              # 현재 플랫폼용 빌드
    python packaging/build.py --vendor-ffmpeg /path/to/ffmpeg /path/to/ffprobe

PyInstaller cannot cross-compile: run this on macOS for the .app and on Windows
for the .exe. Output lands in `dist/`.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENDOR = os.path.join(HERE, "vendor", "bin")


def vendor(paths):
    """Copy ffmpeg/ffprobe into the build so the app ships self-contained."""
    os.makedirs(VENDOR, exist_ok=True)
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"파일이 없습니다: {p}")
        dst = os.path.join(VENDOR, os.path.basename(p))
        shutil.copy2(p, dst)
        os.chmod(dst, 0o755)
        print(f"  번들 포함: {os.path.basename(p)}")
    print("\n⚠️  libx264 가 포함된 ffmpeg 빌드는 GPL 입니다. 배포한다면 해당\n"
          "   라이선스 전문과 소스 제공 의무를 함께 지켜야 합니다.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-ffmpeg", nargs="+", metavar="BIN",
                    help="번들에 포함할 ffmpeg/ffprobe 경로")
    ap.add_argument("--clean", action="store_true", help="build/ dist/ 먼저 삭제")
    a = ap.parse_args()

    if a.vendor_ffmpeg:
        vendor(a.vendor_ffmpeg)
    if a.clean:
        for d in ("build", "dist"):
            shutil.rmtree(os.path.join(ROOT, d), ignore_errors=True)

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm",
           os.path.join(HERE, "autopitch.spec")]
    print("빌드 중:", platform.system(), platform.machine())
    subprocess.run(cmd, cwd=ROOT, check=True)

    dist = os.path.join(ROOT, "dist")
    made = [n for n in os.listdir(dist)] if os.path.isdir(dist) else []
    print("\n완료 →", dist)
    for n in made:
        p = os.path.join(dist, n)
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) \
            else os.path.getsize(p)
        print(f"  {n}  ({sz/1e6:.0f} MB)")
    if not os.path.isdir(VENDOR):
        print("\nffmpeg 미포함 빌드입니다 — 앱이 PATH 에서 찾고, 없으면\n"
              "첫 실행 화면이 설치 방법을 안내합니다.")


if __name__ == "__main__":
    main()
