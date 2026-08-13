# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the macOS .app and the Windows .exe.

Build with `python packaging/build.py` rather than calling pyinstaller directly
-- that script handles the per-platform output naming and the optional ffmpeg
vendoring below.

ffmpeg is bundled ONLY when `packaging/vendor/bin/` contains the binaries. The
build works either way: without them the app resolves ffmpeg from PATH and the
first-run screen tells the user how to install it. Bundling a GPL ffmpeg build
(one compiled with libx264) obliges you to ship its licence and source offer,
so that stays a deliberate, opt-in step rather than a silent default.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
VENDOR = os.path.join(SPECPATH, "vendor", "bin")
# Regenerate both from packaging/icon/icon.png with packaging/make_icons.py.
ICON = os.path.join(SPECPATH, "icon",
                    "autoPitch.icns" if sys.platform == "darwin" else "autoPitch.ico")

datas = [
    (os.path.join(ROOT, "gui", "app.html"), "gui"),
    (os.path.join(ROOT, "config.yaml"), "."),
    (os.path.join(ROOT, "editor"), "editor"),
    (os.path.join(ROOT, "LICENSE"), "."),
]

binaries = []
if os.path.isdir(VENDOR):
    for name in os.listdir(VENDOR):
        binaries.append((os.path.join(VENDOR, name), "bin"))

a = Analysis(
    [os.path.join(ROOT, "autopitch.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=["gui.server", "src.preflight", "src.audio", "src.ffmpeg"],
    # Never freeze the heavy scientific stack we deliberately removed, nor the
    # graphics toolchain (docs-only) -- they add hundreds of MB for nothing.
    excludes=["librosa", "numba", "llvmlite", "sklearn", "scikit_learn",
              "soundfile", "matplotlib", "torch", "torchvision", "ultralytics",
              "cv2", "pandas", "IPython", "tkinter.test", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="autoPitch",
    console=(sys.platform == "win32"),   # Windows keeps a console for the URL
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="autoPitch")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="autoPitch.app",
        icon=ICON,
        bundle_identifier="dev.autopitch.studio",
        info_plist={
            "CFBundleName": "autoPitch",
            "CFBundleDisplayName": "autoPitch",
            "NSHighResolutionCapable": True,
            # No camera/mic use: the app only reads files the user points it at.
            "LSMinimumSystemVersion": "12.0",
        },
    )
